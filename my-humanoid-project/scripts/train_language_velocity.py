"""Train the language-ON G1 (velocity-command following). See PLANS/LANGUAGE_ON_PLAN.md.

Usage (inside the Isaac Lab container, GPU):
    python scripts/train_language_velocity.py --headless --num_envs 4096 --max_iterations 3000

Launches the sim app first, builds the G1LanguageVelocity env, installs the per-episode
command buffers, and trains with RSL-RL PPO. VERIFY ON GPU: RSL-RL import path + env build.
"""
from __future__ import annotations

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--experiment_dir", default="runs/language_velocity")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless or True).app

    from rsl_rl.runners import OnPolicyRunner                       # VERIFY import path
    from isaaclab.envs import ManagerBasedRLEnv
    from my_humanoid_project.tasks.g1_language_velocity_cfg import (
        G1LanguageVelocityEnvCfg, install_command_buffers)

    cfg = G1LanguageVelocityEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg)
    install_command_buffers(env)                                   # randomized cmd per episode

    runner_cfg = {
        "num_steps_per_env": 24, "max_iterations": args.max_iterations, "save_interval": 100,
        "experiment_name": "language_velocity",
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [256, 256, 128],
                   "critic_hidden_dims": [256, 256, 128], "activation": "elu"},
        "algorithm": {"class_name": "PPO", "clip_param": 0.2, "entropy_coef": 0.005,
                      "learning_rate": 5e-4, "schedule": "adaptive", "desired_kl": 0.01,
                      "gamma": 0.99, "lam": 0.95, "num_learning_epochs": 5,
                      "num_mini_batches": 4},
    }
    os.makedirs(args.experiment_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=args.experiment_dir, device=str(env.device))
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(args.experiment_dir, "model_latest.pt"))
    app.close()


if __name__ == "__main__":
    main()
