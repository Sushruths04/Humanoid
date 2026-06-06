"""T2 — Collect GR00T rollouts from LIBERO for world-model training.

Runs the pre-trained GR00T policy on libero_spatial (all 10 tasks) and saves
(obs_state, action, reward) per-step as a .pt dataset compatible with
programs/world_model/train_wm_isaac.py.

obs_state: eef_xyz(3) + axis_angle_rpy(3) + gripper_qpos(2) = 8-dim flat vector
action:    7-dim OSC delta (same as what GR00T outputs)
reward:    0.0 per step, 1.0 at the terminal step if task succeeded

Usage:
    MUJOCO_GL=egl python -m programs.t2_manip_wm.collect_manip_rollouts \
        --checkpoint programs/checkpoints/groot_n17/libero_spatial/libero_spatial \
        --num-episodes 200 \
        --out programs/data/manip_rollouts_groot.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--embodiment-tag", type=str, default="LIBERO_PANDA")
    p.add_argument("--task", type=str, default="libero_spatial")
    p.add_argument("--num-episodes", type=int, default=200,
                   help="Total episodes across all tasks (split evenly)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", type=str, default="programs/data/manip_rollouts_groot.pt")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _extract_state(obs_dict: dict) -> np.ndarray:
    """Extract 8-dim state vector from LIBERO obs dict (same convention as groot_policy.py)."""
    import math
    eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(obs_dict.get("robot0_eef_quat", np.array([0., 0., 0., 1.])), dtype=np.float32)
    gripper = np.asarray(obs_dict.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)

    # quat2axisangle matching groot_policy.py / libero_env.py
    den = np.sqrt(1.0 - float(eef_quat[3]) ** 2)
    if math.isclose(den, 0.0, abs_tol=1e-7):
        rpy = np.zeros(3, dtype=np.float32)
    else:
        rpy = ((eef_quat[:3] * 2.0 * math.acos(float(eef_quat[3]))) / den).astype(np.float32)

    return np.concatenate([eef_pos, rpy, gripper])   # (8,)


def _build_env(bench_name: str, task_idx: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bd = benchmark.get_benchmark_dict()
    b = bd[bench_name]()
    bddl_path = b.get_task_bddl_file_path(task_idx)
    task_obj = b.get_task(task_idx)
    task_language = getattr(task_obj, "language", task_obj.name.replace("_", " "))
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    return env, task_language


def _run_episode(env, policy_fn, max_steps: int):
    obs_list, act_list, rew_list = [], [], []
    obs_dict = env.reset()
    success = False
    for _ in range(max_steps):
        state = _extract_state(obs_dict)
        action = policy_fn(obs_dict)
        obs_dict, _rew, done, _info = env.step(action)
        obs_list.append(state)
        act_list.append(action.astype(np.float32))
        rew_list.append(0.0)
        if env.check_success():
            success = True
        if done or success:
            break
    if rew_list:
        rew_list[-1] = 1.0 if success else 0.0
    return (np.stack(obs_list),        # (T, 8)
            np.stack(act_list),         # (T, 7)
            np.array(rew_list, dtype=np.float32),  # (T,)
            success)


def main():
    args = _parse_args()

    from programs.t1_groot_lora.groot_policy import load_groot_model, make_policy_fn
    print("[t2] loading GR00T model...")
    model = load_groot_model(args.checkpoint, args.embodiment_tag, args.device)

    if ":" in args.task:
        bench_name, idx_str = args.task.split(":", 1)
        task_indices = [int(idx_str)]
    else:
        bench_name = args.task
        task_indices = list(range(10))

    eps_per_task = max(1, args.num_episodes // len(task_indices))
    total_target = eps_per_task * len(task_indices)
    print(f"[t2] collecting {eps_per_task} eps × {len(task_indices)} tasks = {total_target} episodes")

    all_obs, all_act, all_rew = [], [], []
    total_success = 0

    for task_idx in task_indices:
        env, task_language = _build_env(bench_name, task_idx)
        policy_fn, _ = make_policy_fn(model, task_language)
        task_success = 0
        for ep in range(eps_per_task):
            obs_arr, act_arr, rew_arr, success = _run_episode(env, policy_fn, args.max_steps)
            all_obs.append(torch.from_numpy(obs_arr))
            all_act.append(torch.from_numpy(act_arr))
            all_rew.append(torch.from_numpy(rew_arr))
            task_success += int(success)
            total_success += int(success)
        env.close()
        print(f"[t2] task {task_idx}: {task_success}/{eps_per_task} success "
              f"({task_success/eps_per_task:.0%})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": all_obs, "action": all_act, "reward": all_rew}, out)
    print(f"[t2] saved {len(all_obs)} episodes → {out}")
    print(f"[t2] overall success rate: {total_success}/{total_target} "
          f"({total_success/total_target:.1%})")
    print(f"[t2] obs_dim={all_obs[0].shape[-1]}  act_dim={all_act[0].shape[-1]}")


if __name__ == "__main__":
    main()
