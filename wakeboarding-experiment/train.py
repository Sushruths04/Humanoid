"""Train the wakeboard-start policy (PLAN.md §8, §14).

Usage (inside the Isaac Lab container, GPU):
    python train.py --config configs/stage1.yaml [--headless] [--num_envs N] [--max_iterations M]

Standard Isaac Lab pattern: launch the simulator app FIRST (AppLauncher), then import the
env + RSL-RL runner. Curriculum + AMP are wired around the RSL-RL training loop.

VERIFY ON GPU: RSL-RL runner class/import path for the installed version; the exact way the
per-iteration callback exposes rollout success rate to the curriculum.
"""
from __future__ import annotations

import argparse
import os

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--experiment_dir", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.num_envs:
        cfg["num_envs"] = args.num_envs
    if args.max_iterations:
        cfg["max_iterations"] = args.max_iterations

    # 1) launch sim app BEFORE importing isaaclab env modules
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless or True)
    simulation_app = app_launcher.app

    # 2) now safe to import env + RL
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper   # adapts ManagerBasedRLEnv -> rsl_rl VecEnv
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg, T_SUCCESS
    from src.curriculum import PullSpeedCurriculum
    from src.rope_model import kmh_to_ms

    # 3) build env
    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = cfg["num_envs"]
    apply_reward_weights(env_cfg, cfg["rewards"])     # set RewTerm weights BEFORE env construction
    env = WakeboardStartEnv(env_cfg)
    env.rope.model = cfg["rope"]["model"]
    env.rope.set_v_pull(kmh_to_ms(cfg["rope"]["v_pull_kmh"]))

    # 4) curriculum
    cur_cfg = cfg.get("curriculum", {})
    curriculum = PullSpeedCurriculum(
        cur_cfg.get("v_pull_levels_kmh", [cfg["rope"]["v_pull_kmh"]]),
        cur_cfg.get("promote_success_rate", 0.6),
        cur_cfg.get("window", 200),
        enabled=cur_cfg.get("enabled", False),
    )

    # 5) RSL-RL runner (wrap env to the rsl_rl VecEnv interface; keep raw `env` for rope/buffers)
    runner_cfg = build_rsl_rl_cfg(cfg)
    exp_dir = os.path.join(args.experiment_dir, cfg["experiment_name"])
    rl_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(rl_env, runner_cfg, log_dir=exp_dir, device=str(env.device))
    if args.resume:
        runner.load(args.resume)

    # 6) train with a per-iteration curriculum callback
    total = cfg["max_iterations"]
    step = max(50, cur_cfg.get("window", 200))
    done = 0
    while done < total:
        n = min(step, total - done)
        runner.learn(num_learning_iterations=n, init_at_random_ep_len=True)
        done += n
        succ = float(getattr(env, "_success_event", torch.zeros(1)).float().mean().item())
        if curriculum.update(succ):
            env.rope.set_v_pull(curriculum.current_ms)
            print(f"[curriculum] advanced -> {curriculum.current_kmh} km/h")
        runner.save(os.path.join(exp_dir, f"model_{done}.pt"))

    runner.save(os.path.join(exp_dir, "model_latest.pt"))
    simulation_app.close()


def apply_reward_weights(env_cfg, weights: dict):
    """Set RewTerm weights on the cfg object (must be called BEFORE env construction)."""
    for name, w in weights.items():
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.weight = float(w)


def build_rsl_rl_cfg(cfg: dict) -> dict:
    # Use Isaac Lab's own rsl_rl config dataclasses so the emitted dict matches the installed
    # rsl_rl schema. This image is rsl-rl >= 4.0: the policy is split into separate actor/critic
    # RslRlMLPModelCfg models with an obs_groups map (see anymal_d/agents/rsl_rl_ppo_cfg.py).
    # Imported here (not at module top): needs the launched app.
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    ppo = cfg["ppo"]
    hidden = list(ppo["policy_hidden"])
    runner = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=ppo["num_steps_per_env"],
        max_iterations=cfg["max_iterations"],
        save_interval=cfg.get("save_interval", 100),
        experiment_name=cfg["experiment_name"],
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=hidden,
            activation=ppo["activation"],
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=RslRlMLPModelCfg(
            hidden_dims=hidden,
            activation=ppo["activation"],
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=ppo["clip_param"],
            entropy_coef=ppo["entropy_coef"],
            num_learning_epochs=ppo["num_learning_epochs"],
            num_mini_batches=ppo["num_mini_batches"],
            learning_rate=ppo["learning_rate"],
            schedule=ppo["schedule"],
            gamma=ppo["gamma"],
            lam=ppo["lam"],
            desired_kl=ppo["desired_kl"],
            max_grad_norm=1.0,
        ),
    )
    rl_cfg = runner.to_dict()
    # rsl-rl >= 5.0 MLPModel uses distribution_cfg only; to_dict() still emits the deprecated
    # scalar fields, which MLPModel.__init__ rejects as unexpected kwargs. Strip them.
    for grp in ("actor", "critic"):
        for dep in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            if isinstance(rl_cfg.get(grp), dict):
                rl_cfg[grp].pop(dep, None)
    return rl_cfg


if __name__ == "__main__":
    main()
