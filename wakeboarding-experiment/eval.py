"""Evaluate a wakeboard checkpoint and write metrics JSON (PLAN.md §10.2, §10.3).

Usage:
    python eval.py --checkpoint runs/.../model_latest.pt --v_pull_kmh 30 --episodes 200 \
                   --out results/eval_30kmh.json
Speed sweep (Table A) is just this over several --v_pull_kmh via 31_eval_speed_sweep.sh.

Metrics: success_rate, fall_rate, mean_time_to_stand, mean_episode_length,
board_angle_adherence, arm_straightness, smoothness, energy.
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
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg, T_SUCCESS
    from src.rope_model import kmh_to_ms

    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = WakeboardStartEnv(env_cfg)
    env.rope.set_v_pull(kmh_to_ms(args.v_pull_kmh))

    runner = OnPolicyRunner(env, _min_cfg(), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    # rollout
    n_succ = n_fall = 0
    t_stand, ep_len, board_adh, arm_str, smooth, energy = [], [], [], [], [], []
    obs, _ = env.reset()
    collected = 0
    ep_steps = torch.zeros(env.num_envs, device=env.device)
    while collected < args.episodes:
        with torch.no_grad():
            act = policy(obs)
        obs, _, dones, _ = env.step(act)
        ep_steps += 1
        board_adh.append(((env._board_pitch > 10 * 3.14159 / 180) &
                          (env._board_pitch < 20 * 3.14159 / 180)).float().mean().item())
        arm_str.append((-env._elbow_flexion).exp().mean().item())
        done_idx = dones.nonzero().flatten()
        for i in done_idx.tolist():
            collected += 1
            if env._success_event[i]:
                n_succ += 1
                t_stand.append(env._stable_time[i].item())
            if env._fall_event[i]:
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
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    app.close()


def _mean(x):
    return sum(x) / len(x) if x else None


def _min_cfg():
    return {"num_steps_per_env": 24, "policy": {"class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
            "activation": "elu"}, "algorithm": {"class_name": "PPO"}}


if __name__ == "__main__":
    main()
