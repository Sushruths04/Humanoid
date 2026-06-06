"""T3 — Evaluate pixel BC policy on LIBERO Spatial (pixel-only obs, no state).

Usage:
    MUJOCO_GL=egl PYTHONUNBUFFERED=1 python -m programs.t3_vision_manip.evaluate_pixel_bc \
        --checkpoint programs/checkpoints/t3_pixel_bc/pixel_bc.pt \
        --task libero_spatial \
        --num-envs 10 \
        --out docs/results/t3_pixel_bc.md
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="libero_spatial",
                   help="'<bench>' for all 10 tasks or '<bench>:<idx>' for one")
    p.add_argument("--num-envs", type=int, default=10, help="Episodes per task")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out", type=str, default="docs/results/t3_pixel_bc.md")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--video-dir", type=str, default="programs/videos/t3_pixel_bc")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def _build_env(bench_name: str, task_idx: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bd = benchmark.get_benchmark_dict()
    b = bd[bench_name]()
    bddl_path = b.get_task_bddl_file_path(task_idx)
    task_obj = b.get_task(task_idx)
    task_name = task_obj.name
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
    )
    print(f"[eval] task {task_idx}: {task_name}")
    return env, task_name


def _run_episode(env, policy_fn, max_steps: int, recorder=None):
    import numpy as np
    obs_dict = env.reset()
    task_success = False
    for step in range(max_steps):
        img = obs_dict["agentview_image"]    # (256, 256, 3) uint8
        if recorder is not None:
            recorder.add_frame(img)
        action = policy_fn(img)
        obs_dict, _rew, done, _info = env.step(action)
        if env.check_success():
            task_success = True
        if done or task_success:
            break
    return task_success


def _load_policy(ckpt_path: str, device: str):
    import torch
    from programs.t3_vision_manip.pixel_bc_policy import PixelBCPolicy, preprocess_image

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PixelBCPolicy(action_dim=ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def policy_fn(img_hwc_uint8):
        with torch.no_grad():
            img_t = preprocess_image(img_hwc_uint8, device=device)
            action = model(img_t).squeeze(0).cpu().numpy()
        return action

    return policy_fn, ckpt


def _write_report(results: list[dict], out: Path, ckpt_meta: dict, args):
    tasks = sorted({r["task"] for r in results})
    lines = [
        "# T3 Pixel BC — Evaluation Results",
        "",
        f"Checkpoint: `{args.checkpoint}`  ",
        f"Obs: agentview_image only (no state)  ",
        f"Episodes per task: {args.num_envs}  ",
        "",
        "## Per-Task Results",
        "",
        "| Task | Success Rate |",
        "|---|---|",
    ]
    total_sr = 0.0
    for task in tasks:
        rows = [r for r in results if r["task"] == task]
        sr = sum(r["success"] for r in rows) / len(rows)
        total_sr += sr
        lines.append(f"| {task.replace('_',' ')} | **{sr:.3f}** |")

    mean_sr = total_sr / len(tasks) if tasks else 0.0
    dod = "PASS ✅" if mean_sr >= 0.20 else "FAIL ❌"
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean task success | **{mean_sr:.3f}** |",
        f"| DoD (>0.20) | {dod} |",
        f"| Training transitions | {ckpt_meta.get('num_transitions','?')} |",
        f"| Training loss | {ckpt_meta.get('init_loss',0):.5f} → {ckpt_meta.get('final_loss',0):.5f} |",
        f"| T0 BC baseline (state-obs) | 0.500 (task 0 only) |",
        f"| T1 GR00T (pixel+state) | 0.970 (all 10 tasks) |",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[eval] report → {out}")
    print(f"[eval] mean_task_success={mean_sr:.3f}  DoD={dod}")


def main():
    args = _parse_args()
    device = args.device

    policy_fn, ckpt_meta = _load_policy(args.checkpoint, device)

    if ":" in args.task:
        bench_name, idx_str = args.task.split(":", 1)
        task_indices = [int(idx_str)]
    else:
        bench_name = args.task
        task_indices = list(range(10))

    from programs.common.eval.video_recorder import EpisodeRecorder
    recorder = None if args.no_video else EpisodeRecorder(
        out_dir=args.video_dir, fps=10, max_per_task=1, record_failures=True
    )

    all_results = []
    for task_idx in task_indices:
        env, task_name = _build_env(bench_name, task_idx)
        for ep in range(args.num_envs):
            if recorder is not None:
                recorder.start_episode(task_name, ep)
            success = _run_episode(env, policy_fn, args.max_steps, recorder)
            if recorder is not None:
                recorder.finish_episode(success)
            all_results.append({"task": task_name, "ep": ep, "success": success})
            print(f"[eval] {task_name} ep{ep:02d}: {'SUCCESS' if success else 'fail'}")
        env.close()

    _write_report(all_results, Path(args.out), ckpt_meta, args)


if __name__ == "__main__":
    main()
