"""Evaluate a manipulation policy on LIBERO tabletop tasks.

Rolls out the policy across N episodes on a single LIBERO task, recording:
  - grasped: robot achieved stable grasp (object lifted above threshold)
  - placed: task success per LIBERO BDDL predicates
  - dropped: object fell significantly after being grasped
  - task_success: same as placed (LIBERO ground truth)
  - steps_to_success: first step at which task_success triggered (-1 if never)

Usage:
    python -m programs.t0_manip_foundation.evaluate_manip \
        --task libero_spatial:0 \
        --num-envs 10 \
        --out docs/results/t0_manip.md

    # With a trained BC checkpoint:
    python -m programs.t0_manip_foundation.evaluate_manip \
        --task libero_spatial:0 \
        --checkpoint programs/checkpoints/t0_bc/bc_libero_spatial_0.pt \
        --num-envs 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate manipulation policy on LIBERO.")
    parser.add_argument("--task", type=str, required=True,
                        help="LIBERO task spec: '<benchmark_name>:<task_idx>' e.g. 'libero_spatial:0'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to BC policy checkpoint (.pt). Omit for random policy.")
    parser.add_argument("--num-envs", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode before timeout")
    parser.add_argument("--grasp-height-threshold", type=float, default=0.03,
                        help="Min lift height (m) above initial object z to count as grasp")
    parser.add_argument("--drop-threshold", type=float, default=0.05,
                        help="Height drop (m) after grasp to count as dropped")
    parser.add_argument("--out", type=str, default="docs/results/t0_manip.md")
    return parser.parse_args()


def _obs_to_flat(obs: dict, compact: bool = False) -> "np.ndarray":
    """Flatten LIBERO obs dict to 1-D array.

    compact=False (default): proprio-state(39) + object-state(70) = 109 dims.
    compact=True: joint_pos(7) + eef_pos(3) + gripper_qpos(2) = 12 dims —
        consistent with HDF5 demo obs used in train_bc_libero.py.
    """
    import numpy as np
    if compact:
        joint = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32)
        eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
        grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
        return np.concatenate([joint, eef, grip])
    parts = []
    for key in ("robot0_proprio-state", "object-state"):
        if key in obs:
            parts.append(np.asarray(obs[key], dtype=np.float32))
    if not parts:
        parts = [np.asarray(v, dtype=np.float32).flatten()
                 for v in obs.values() if hasattr(v, "__len__") and len(v) < 256]
    return np.concatenate(parts) if parts else np.zeros(109, dtype=np.float32)


def _run_episode(env, policy_fn, max_steps: int,
                 grasp_h_thresh: float, drop_thresh: float,
                 compact_obs: bool = False):
    """Run one episode; return (grasped, placed, dropped, task_success, steps_to_success)."""
    import numpy as np

    obs_dict = env.reset()
    obs_flat = _obs_to_flat(obs_dict, compact=compact_obs)

    # Initial object z from object-state[2] (z coord of first object)
    obj_state0 = obs_dict.get("object-state", np.zeros(70))
    initial_obj_z = float(obj_state0[2])

    did_grasp = False
    did_drop = False
    task_success = False
    max_obj_z_after_grasp = initial_obj_z
    success_step = -1

    for step in range(max_steps):
        action = policy_fn(obs_flat)
        obs_dict, _reward, done, _info = env.step(action)
        obs_flat = _obs_to_flat(obs_dict, compact=compact_obs)

        obj_state = obs_dict.get("object-state", np.zeros(70))
        obj_z = float(obj_state[2])

        # Grasp: object lifted above initial position by threshold
        if not did_grasp and obj_z > initial_obj_z + grasp_h_thresh:
            did_grasp = True
            max_obj_z_after_grasp = obj_z

        if did_grasp:
            max_obj_z_after_grasp = max(max_obj_z_after_grasp, obj_z)
            # Drop: fell more than drop_thresh below the highest point
            if not did_drop and (max_obj_z_after_grasp - obj_z) > drop_thresh:
                did_drop = True

        # Task success via LIBERO BDDL predicates
        if env.check_success():
            task_success = True
            if success_step < 0:
                success_step = step

        if done or task_success:
            break

    did_place = task_success
    return did_grasp, did_place, did_drop, task_success, success_step


def _build_env(args):
    """Build LIBERO OffScreenRenderEnv for the given task spec."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench_name, task_idx_str = (args.task.split(":", 1) if ":" in args.task
                                else (args.task, "0"))
    task_idx = int(task_idx_str)

    bd = benchmark.get_benchmark_dict()
    if bench_name not in bd:
        raise ValueError(f"Unknown LIBERO benchmark '{bench_name}'. "
                         f"Valid: {list(bd.keys())}")

    b = bd[bench_name]()
    bddl_path = b.get_task_bddl_file_path(task_idx)
    task_name = b.get_task(task_idx).name
    print(f"[eval] task: {task_name}")
    print(f"[eval] bddl: {bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    return env


def _build_policy(args):
    """Load BC policy from checkpoint, or return a random policy for smoke-tests."""
    import numpy as np
    import torch

    act_dim = 7  # LIBERO uses OSC_POSE: 3 pos + 3 rot + 1 gripper

    if args.checkpoint:
        from programs.t0_manip_foundation.bc_baseline import MLPBCPolicy
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        obs_dim = ckpt.get("obs_dim", 109)
        action_dim = ckpt.get("action_dim", act_dim)
        hidden = ckpt.get("hidden", 256)
        compact = ckpt.get("obs_mode", "") == "compact"
        pol = MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden)
        pol.net.load_state_dict(ckpt["model_state"])
        pol.net.eval()

        def policy_fn(obs_flat: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
                return pol.forward(t).squeeze(0).numpy()

        print(f"[eval] loaded BC policy from {args.checkpoint} (obs={obs_dim} act={action_dim} compact={compact})")
    else:
        rng = np.random.default_rng(42)

        def policy_fn(obs_flat: np.ndarray) -> np.ndarray:
            return rng.uniform(-0.03, 0.03, act_dim).astype(np.float32)

        print("[eval] using random policy (no checkpoint given)")
        compact = False

    return policy_fn, compact


def main():
    args = _parse_args()
    import torch
    from programs.common.eval.manip_metrics import compute_manip_metrics

    print(f"[eval] building env for task={args.task}")
    env = _build_env(args)
    policy_fn, compact_obs = _build_policy(args)

    grasped_l, placed_l, dropped_l, success_l, steps_l = [], [], [], [], []

    for ep in range(args.num_envs):
        g, p, d, t, s = _run_episode(
            env, policy_fn,
            max_steps=args.max_steps,
            grasp_h_thresh=args.grasp_height_threshold,
            drop_thresh=args.drop_threshold,
            compact_obs=compact_obs,
        )
        grasped_l.append(g)
        placed_l.append(p)
        dropped_l.append(d)
        success_l.append(t)
        steps_l.append(s)
        if (ep + 1) % max(1, args.num_envs // 5) == 0:
            print(f"[eval] episode {ep+1}/{args.num_envs}  success={sum(success_l)}/{ep+1}")

    metrics = compute_manip_metrics(
        torch.tensor(grasped_l, dtype=torch.float32),
        torch.tensor(placed_l, dtype=torch.float32),
        torch.tensor(dropped_l, dtype=torch.float32),
        torch.tensor(success_l, dtype=torch.float32),
        torch.tensor(steps_l, dtype=torch.long),
    )
    metrics["checkpoint"] = str(args.checkpoint or "random")
    metrics["task"] = args.task

    print(f"[eval] results: {metrics}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(metrics, out, args)
    print(f"[eval] report written to {out}")

    env.close()


def _write_report(metrics: dict, out: Path, args) -> None:
    lines = [
        f"# T0 Manipulation Eval: {args.task}",
        "",
        f"Policy: `{args.checkpoint or 'random'}`  ",
        f"Episodes: {metrics['num_episodes']}  ",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| task_success | **{metrics['task_success']:.3f}** |",
        f"| grasp_success | {metrics['grasp_success']:.3f} |",
        f"| place_success | {metrics['place_success']:.3f} |",
        f"| object_drop_rate | {metrics['object_drop_rate']:.3f} |",
        f"| mean_steps_to_success | {metrics['mean_steps_to_success']:.1f} |",
        "",
    ]
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
