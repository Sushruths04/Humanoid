"""Evaluate a manipulation policy on LIBERO or Isaac Lab Franka tasks.

Rolls out the policy across N envs, recording per-episode:
  - grasped: robot achieved stable grasp
  - placed: robot placed object at target
  - dropped: robot dropped object after grasping
  - task_success: full task done (grasped + placed, no drop)
  - steps_to_success: first step at which task_success triggered (-1 if never)

Writes a markdown report via programs.common.eval.report and returns metrics.

Usage:
    python -m programs.t0_manip_foundation.evaluate_manip \
        --task <LIBERO_TASK_NAME> \
        --checkpoint <path/to/policy.pt> \
        --num-envs 64 \
        --out docs/results/t0_manip.md
"""

from __future__ import annotations

import argparse
import os
import math
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate manipulation policy.")
    parser.add_argument("--task", type=str, required=True,
                        help="LIBERO task name or Isaac Lab env ID")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--grasp-height-threshold", type=float, default=0.05,
                        help="Min lift height (m) above table to count as grasp")
    parser.add_argument("--place-radius", type=float, default=0.05,
                        help="Max distance (m) to target pos to count as placed")
    parser.add_argument("--drop-threshold", type=float, default=0.05,
                        help="Height drop (m) after grasp to count as dropped")
    parser.add_argument("--out", type=str, default="docs/results/t0_manip.md")
    return parser.parse_args()


def _libero_episode(env, policy, max_steps: int, grasp_h: float, place_r: float, drop_t: float):
    """Roll out one episode; return (grasped, placed, dropped, task_success, steps)."""
    import torch
    obs = env.reset()
    did_grasp = False
    did_place = False
    did_drop = False
    grasp_height = None
    min_h_after = float("inf")
    success_step = -1

    for step in range(max_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, reward, done, info = env.step(action)

        obj_h = float(info.get("object_height", 0.0))
        obj_dist_to_target = float(info.get("object_dist_to_target", float("inf")))
        grasping_now = obj_h > grasp_h

        if grasping_now and not did_grasp:
            did_grasp = True
            grasp_height = obj_h

        if did_grasp:
            min_h_after = min(min_h_after, obj_h)
            if not did_place and obj_dist_to_target < place_r:
                did_place = True

        if did_grasp and (grasp_height - min_h_after) > drop_t:
            did_drop = True

        task_done = did_grasp and did_place and not did_drop
        if task_done and success_step < 0:
            success_step = step

        if done:
            break

    return did_grasp, did_place, did_drop, (did_grasp and did_place and not did_drop), success_step


def main():
    args = _parse_args()

    try:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args)
        sim_app = app_launcher.app
    except ImportError:
        sim_app = None

    import torch
    from programs.common.eval.manip_metrics import compute_manip_metrics
    from programs.common.eval.report import write_results_markdown

    # Env + policy construction happens here; imports guarded by try/except
    # so this module is importable on CPU for testing.
    try:
        env = _build_env(args)
        policy = _load_policy(args, env)
    except Exception as exc:
        print(f"[eval] Cannot build env/policy: {exc}")
        if sim_app:
            sim_app.close()
        return

    grasped_list, placed_list, dropped_list, success_list, steps_list = [], [], [], [], []
    for _ in range(args.num_envs):
        g, p, d, t, s = _libero_episode(
            env, policy, args.max_steps,
            args.grasp_height_threshold, args.place_radius, args.drop_threshold,
        )
        grasped_list.append(g)
        placed_list.append(p)
        dropped_list.append(d)
        success_list.append(t)
        steps_list.append(s)

    metrics = compute_manip_metrics(
        torch.tensor(grasped_list),
        torch.tensor(placed_list),
        torch.tensor(dropped_list),
        torch.tensor(success_list),
        torch.tensor(steps_list, dtype=torch.long),
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["task"] = args.task

    print("[eval] metrics:", metrics)
    write_results_markdown(metrics, args.out, title=f"Manipulation Eval: {args.task}")
    print(f"[eval] wrote {args.out}")

    env.close()
    if sim_app:
        sim_app.close()


def _build_env(args):
    """Build LIBERO or Isaac Lab manipulation env — extend for your framework."""
    raise NotImplementedError(
        "Wire in your LIBERO / Isaac Lab env here. "
        "See programs/t0_manip_foundation/README.md for setup instructions."
    )


def _load_policy(args, env):
    """Load policy checkpoint — extend for your framework."""
    raise NotImplementedError(
        "Wire in your policy loading here (LeRobot ACT/DP or custom)."
    )


if __name__ == "__main__":
    main()
