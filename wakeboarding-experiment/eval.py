"""Evaluate a wakeboard checkpoint and write metrics JSON (PLAN.md §10.2, §10.3).

Usage:
    python eval.py --checkpoint runs/.../model_latest.pt --v_pull_kmh 30 --episodes 200 \
                   --out results/eval_30kmh.json
"""
from __future__ import annotations

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--v_pull_kmh", type=float, default=30.0)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg, T_SUCCESS
    from src.rope_model import kmh_to_ms

    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    base_env = WakeboardStartEnv(env_cfg)
    base_env.rope.set_v_pull(kmh_to_ms(args.v_pull_kmh))
    env = RslRlVecEnvWrapper(base_env)

    runner = OnPolicyRunner(env, _min_cfg(), log_dir=None, device=str(base_env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    # rollout
    n_succ = n_fall = 0
    t_stand, ep_len, board_adh, arm_str = [], [], [], []
    obs, _ = env.reset()
    collected = 0
    ep_steps = torch.zeros(base_env.num_envs, device=base_env.device)
    while collected < args.episodes:
        with torch.no_grad():
            act = policy(obs)
        result = env.step(act)
        obs, _, dones = result[0], result[1], result[2]
        ep_steps += 1
        board_adh.append(((base_env._board_pitch > 10 * 3.14159 / 180) &
                          (base_env._board_pitch < 20 * 3.14159 / 180)).float().mean().item())
        arm_str.append((-base_env._elbow_flexion).exp().mean().item())
        done_idx = dones.nonzero(as_tuple=False).flatten()
        for i in done_idx.tolist():
            collected += 1
            if base_env._success_event[i]:
                n_succ += 1
                t_stand.append(base_env._stable_time[i].item())
            if base_env._fall_event[i]:
                n_fall += 1
            ep_len.append(ep_steps[i].item())
            ep_steps[i] = 0
            if collected >= args.episodes:
                break

    res = {
        "checkpoint": args.checkpoint,
        "v_pull_kmh": args.v_pull_kmh,
        "episodes": collected,
        "success_rate": n_succ / max(collected, 1),
        "fall_rate": n_fall / max(collected, 1),
        "mean_time_to_stand_s": _mean(t_stand),
        "mean_episode_length": _mean(ep_len),
        "board_angle_adherence": _mean(board_adh),
        "arm_straightness": _mean(arm_str),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    app.close()


def _mean(x):
    return sum(x) / len(x) if x else None


def _min_cfg():
    hidden = [256, 256]
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    runner = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24, max_iterations=1, save_interval=1,
        experiment_name="eval",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=hidden, activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=hidden, activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=5e-4, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )
    rl_cfg = runner.to_dict()
    for grp in ("actor", "critic"):
        for dep in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            if isinstance(rl_cfg.get(grp), dict):
                rl_cfg[grp].pop(dep, None)
    return rl_cfg


if __name__ == "__main__":
    main()
