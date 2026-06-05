"""T1 — Evaluate GR00T N1.7 on LIBERO using our existing eval harness.

Runs GR00T in the same LIBERO env used by T0, records metrics in the same
format, and writes docs/results/t1_groot.md. The policy is called at 5 Hz
via the synchronous step loop (no client-server needed for single-GPU eval).

Hardware: 1× GPU ≥16 GB VRAM (L40S or better). T4 (15 GB) may work if
          --denoising-steps 4 is set (reduces memory usage).

Usage:
    MUJOCO_GL=egl python -m programs.t1_groot_lora.evaluate_groot_libero \
        --checkpoint programs/checkpoints/groot_n17/libero_spatial/libero_spatial \
        --task libero_spatial:0 \
        --num-envs 50 \
        --out docs/results/t1_groot.md

    # All 10 spatial tasks at once:
    MUJOCO_GL=egl python -m programs.t1_groot_lora.evaluate_groot_libero \
        --checkpoint programs/checkpoints/groot_n17/libero_spatial/libero_spatial \
        --task libero_spatial \
        --num-envs 20 \
        --out docs/results/t1_groot.md
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
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to GR00T model dir (config.json + safetensors)")
    p.add_argument("--embodiment-tag", type=str, default="LIBERO_PANDA")
    p.add_argument("--task", type=str, default="libero_spatial",
                   help="'<bench>:<idx>' for single task or '<bench>' for all tasks")
    p.add_argument("--num-envs", type=int, default=20,
                   help="Episodes per task")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--denoising-steps", type=int, default=None,
                   help="Reduce for lower VRAM (default: model default ~10-20)")
    p.add_argument("--action-horizon", type=int, default=8,
                   help="Actions per GR00T inference call (chunk size)")
    p.add_argument("--out", type=str, default="docs/results/t1_groot.md")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _build_libero_env(bench_name: str, task_idx: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bd = benchmark.get_benchmark_dict()
    b = bd[bench_name]()
    bddl_path = b.get_task_bddl_file_path(task_idx)
    task_name = b.get_task(task_idx).name
    print(f"[eval] task {task_idx}: {task_name}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
    )
    return env, task_name


def _run_episode_groot(env, policy_fn, max_steps: int,
                        grasp_h_thresh: float = 0.03, drop_thresh: float = 0.05):
    """Run one LIBERO episode with a GR00T policy (obs_dict → action).
    Returns (grasped, placed, dropped, task_success, steps_to_success).
    """
    import numpy as np

    obs_dict = env.reset()
    obj_state0 = obs_dict.get("object-state", np.zeros(70))
    initial_obj_z = float(obj_state0[2])

    did_grasp = did_drop = task_success = False
    max_obj_z = initial_obj_z
    success_step = -1

    step = 0
    while step < max_steps:
        # GR00T outputs an action chunk; we execute action_horizon steps before re-querying
        action = policy_fn(obs_dict)
        obs_dict, _rew, done, _info = env.step(action)

        obj_state = obs_dict.get("object-state", np.zeros(70))
        obj_z = float(obj_state[2])

        if not did_grasp and obj_z > initial_obj_z + grasp_h_thresh:
            did_grasp = True
        if did_grasp:
            max_obj_z = max(max_obj_z, obj_z)
            if not did_drop and (max_obj_z - obj_z) > drop_thresh:
                did_drop = True

        if env.check_success():
            task_success = True
            if success_step < 0:
                success_step = step

        step += 1
        if done or task_success:
            break

    return did_grasp, task_success, did_drop, task_success, success_step


def _write_report(all_results: list[dict], out: Path, args):
    import json

    tasks = sorted({r["task"] for r in all_results})
    lines = [
        "# T1 GR00T N1.7 Eval: LIBERO Spatial",
        "",
        f"Checkpoint: `{args.checkpoint}`  ",
        f"Embodiment: `{args.embodiment_tag}`  ",
        f"Episodes per task: {args.num_envs}  ",
        "",
        "## Per-Task Results",
        "",
        "| Task | Success Rate | Grasp Rate |",
        "|---|---|---|",
    ]
    total_success = 0
    for task in tasks:
        rows = [r for r in all_results if r["task"] == task]
        sr = sum(r["task_success"] for r in rows) / len(rows)
        gr = sum(r["grasped"] for r in rows) / len(rows)
        total_success += sr
        lines.append(f"| {task.replace('_',' ')} | **{sr:.3f}** | {gr:.3f} |")

    mean_sr = total_success / len(tasks) if tasks else 0.0
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mean_task_success | **{mean_sr:.3f}** |",
        f"| T0 BC baseline | 0.500 (single task) |",
        f"| GR00T N1.7 (official NVIDIA) | ~0.977 (200-ep benchmark) |",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[eval] report → {out}")


def main():
    args = _parse_args()

    # Build GR00T policy
    from programs.t1_groot_lora.groot_policy import build_groot_policy
    policy_fn, _ = build_groot_policy(
        model_path=args.checkpoint,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        action_horizon=args.action_horizon,
        denoising_steps=args.denoising_steps,
    )

    import torch
    from programs.common.eval.manip_metrics import compute_manip_metrics

    # Parse task spec: "libero_spatial" → all 10 tasks; "libero_spatial:0" → just task 0
    if ":" in args.task:
        bench_name, idx_str = args.task.split(":", 1)
        task_indices = [int(idx_str)]
    else:
        bench_name = args.task
        task_indices = list(range(10))  # libero_spatial has 10 tasks

    all_results = []

    for task_idx in task_indices:
        env, task_name = _build_libero_env(bench_name, task_idx)

        for ep in range(args.num_envs):
            g, p, d, t, s = _run_episode_groot(env, policy_fn, args.max_steps)
            all_results.append({
                "task": task_name,
                "grasped": int(g),
                "placed": int(p),
                "dropped": int(d),
                "task_success": int(t),
                "steps": s,
            })
            if (ep + 1) % max(1, args.num_envs // 5) == 0:
                recent = all_results[-args.num_envs:]
                sr = sum(r["task_success"] for r in recent) / len(recent)
                print(f"  [{task_name[:40]}] ep {ep+1}/{args.num_envs}  sr={sr:.2f}")

        env.close()

    _write_report(all_results, Path(args.out), args)
    print("[eval] done")


if __name__ == "__main__":
    main()
