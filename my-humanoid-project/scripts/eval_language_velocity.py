"""Evaluate the language-ON G1: prove distinct commands -> distinct behavior.

Produces:
  - eval_language.json : per-command tracking error + a 'separation_score'
  - behavior_separation.png : commanded vs achieved (vx, yaw) bar chart per command

Usage:
    python scripts/eval_language_velocity.py --checkpoint runs/language_velocity/model_latest.pt \
        --out_json results/eval_language.json --out_png results/behavior_separation.png

This is the acceptance test from PLANS/LANGUAGE_ON_PLAN.md: if the bars differ per command
(stand≈0, fast>slow>0, turn-left/right opposite yaw), language is genuinely ON.
"""
from __future__ import annotations

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out_json", default="results/eval_language.json")
    p.add_argument("--out_png", default="results/behavior_separation.png")
    return p.parse_args()


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import ManagerBasedRLEnv
    from my_humanoid_project.tasks.g1_language_velocity_cfg import (
        G1LanguageVelocityEnvCfg, install_command_buffers)
    from my_humanoid_project.language_velocity_commands import VEL_COMMANDS, NUM_COMMANDS

    cfg = G1LanguageVelocityEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg)
    install_command_buffers(env)
    runner = OnPolicyRunner(env, _min_cfg(), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    per_cmd = {}
    for c in VEL_COMMANDS:
        env._cmd_ids[:] = c.cmd_id                       # force ALL envs to this command
        obs, _ = env.reset()
        env._cmd_ids[:] = c.cmd_id                       # reset may resample; pin again
        vx = vy = yaw = 0.0
        for _ in range(args.steps):
            with torch.no_grad():
                a = policy(obs)
            obs, _, _, _ = env.step(a)
            d = env.scene["robot"].data
            vx += d.root_lin_vel_b[:, 0].mean().item()
            vy += d.root_lin_vel_b[:, 1].mean().item()
            yaw += d.root_ang_vel_b[:, 2].mean().item()
        n = args.steps
        achieved = {"vx": vx / n, "vy": vy / n, "yaw": yaw / n}
        err = ((achieved["vx"] - c.vx) ** 2 + (achieved["vy"] - c.vy) ** 2 +
               (achieved["yaw"] - c.yaw) ** 2) ** 0.5
        per_cmd[c.text] = {"target": {"vx": c.vx, "vy": c.vy, "yaw": c.yaw},
                           "achieved": achieved, "track_error": err}

    # separation score: mean over commands of |achieved - other commands' achieved|, normalized
    sep = _separation_score(per_cmd)
    res = {"checkpoint": args.checkpoint, "per_command": per_cmd, "separation_score": sep,
           "language_is_on": sep > 0.3}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(res, open(args.out_json, "w"), indent=2)
    _plot(per_cmd, args.out_png)
    print(json.dumps(res, indent=2))
    print(f"\nlanguage_is_on = {res['language_is_on']} (separation_score={sep:.3f})")
    app.close()


def _separation_score(per_cmd):
    import itertools
    vxs = [v["achieved"]["vx"] for v in per_cmd.values()]
    yaws = [v["achieved"]["yaw"] for v in per_cmd.values()]
    diffs = [abs(a - b) for a, b in itertools.combinations(vxs, 2)]
    diffs += [abs(a - b) for a, b in itertools.combinations(yaws, 2)]
    return sum(diffs) / len(diffs) if diffs else 0.0


def _plot(per_cmd, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(per_cmd.keys())
        tgt_vx = [per_cmd[n]["target"]["vx"] for n in names]
        ach_vx = [per_cmd[n]["achieved"]["vx"] for n in names]
        tgt_yaw = [per_cmd[n]["target"]["yaw"] for n in names]
        ach_yaw = [per_cmd[n]["achieved"]["yaw"] for n in names]
        x = range(len(names))
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        ax[0].bar([i - 0.2 for i in x], tgt_vx, 0.4, label="target vx")
        ax[0].bar([i + 0.2 for i in x], ach_vx, 0.4, label="achieved vx")
        ax[0].set_ylabel("forward speed (m/s)"); ax[0].legend(); ax[0].set_xticks(list(x))
        ax[0].set_xticklabels(names, rotation=20)
        ax[1].bar([i - 0.2 for i in x], tgt_yaw, 0.4, label="target yaw")
        ax[1].bar([i + 0.2 for i in x], ach_yaw, 0.4, label="achieved yaw")
        ax[1].set_ylabel("yaw rate (rad/s)"); ax[1].legend(); ax[1].set_xticks(list(x))
        ax[1].set_xticklabels(names, rotation=20)
        fig.suptitle("Language-ON: commanded vs achieved behavior per command")
        fig.tight_layout(); fig.savefig(path, dpi=120)
        print(f"saved {path}")
    except Exception as e:
        print(f"plot skipped ({e})")


def _min_cfg():
    return {"num_steps_per_env": 24, "policy": {"class_name": "ActorCritic",
            "actor_hidden_dims": [256, 256, 128], "critic_hidden_dims": [256, 256, 128],
            "activation": "elu"}, "algorithm": {"class_name": "PPO"}}


if __name__ == "__main__":
    main()
