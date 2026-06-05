"""Train a BC policy directly from LIBERO HDF5 demonstration files.

Loads state-only obs: joint_pos(7) + eef_pos(3) + gripper_qpos(2) = 12 dims.
Action = 7-dim OSC_POSE delta (from demo["actions"]).

The same 12-dim obs can be extracted from the LIBERO env at inference time via
the --obs-mode compact flag in evaluate_manip.py.

Usage:
    python -m programs.t0_manip_foundation.train_bc_libero \
        --data-dir /teamspace/studios/this_studio/libero_datasets/libero_spatial \
        --task-idx 0 \
        --epochs 200 \
        --out programs/checkpoints/t0_bc/bc_libero_spatial_0.pt
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing libero_spatial/*.hdf5 demo files")
    p.add_argument("--task-idx", type=int, default=0,
                   help="Which task (0-9 for libero_spatial) to train on")
    p.add_argument("--max-demos", type=int, default=50,
                   help="Max demonstrations to use per task file")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", type=str,
                   default="programs/checkpoints/t0_bc/bc_libero_spatial_0.pt")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _load_hdf5_demos(hdf5_path: str, max_demos: int) -> tuple[np.ndarray, np.ndarray]:
    """Load (obs, action) pairs from a LIBERO HDF5 demo file.

    Obs = joint_states(7) + ee_pos(3) + gripper_states(2) = 12 dims.
    Action = 7-dim OSC_POSE.
    """
    import h5py

    all_obs, all_acts = [], []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())[:max_demos]
        for dk in demo_keys:
            demo = f["data"][dk]
            joint = np.array(demo["obs"]["joint_states"])       # (T, 7)
            eef = np.array(demo["obs"]["ee_states"])[:, :3]     # (T, 3) pos only
            grip = np.array(demo["obs"]["gripper_states"])      # (T, 2)
            acts = np.array(demo["actions"])                    # (T, 7)
            obs = np.concatenate([joint, eef, grip], axis=1)   # (T, 12)
            # Skip last frame (no action for it)
            all_obs.append(obs[:-1])
            all_acts.append(acts[:-1])

    obs_arr = np.concatenate(all_obs, axis=0).astype(np.float32)
    act_arr = np.concatenate(all_acts, axis=0).astype(np.float32)
    return obs_arr, act_arr


def _task_hdf5_path(data_dir: str, task_idx: int) -> str:
    """Get the HDF5 file path for a given task index in libero_spatial."""
    import h5py

    files = sorted(Path(data_dir).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    if task_idx >= len(files):
        raise IndexError(f"task_idx={task_idx} but only {len(files)} files in {data_dir}")
    return str(files[task_idx])


def _build_model(obs_dim: int, action_dim: int, hidden: int):
    from programs.t0_manip_foundation.bc_baseline import MLPBCPolicy
    return MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden)


def main():
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hdf5_path = _task_hdf5_path(args.data_dir, args.task_idx)
    print(f"[bc] loading {Path(hdf5_path).name}  max_demos={args.max_demos}")
    obs_arr, act_arr = _load_hdf5_demos(hdf5_path, args.max_demos)
    print(f"[bc] dataset: {obs_arr.shape[0]} transitions  obs={obs_arr.shape[1]}  act={act_arr.shape[1]}")

    obs_t = torch.from_numpy(obs_arr)
    act_t = torch.from_numpy(act_arr)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    obs_dim, action_dim = obs_arr.shape[1], act_arr.shape[1]
    model = _build_model(obs_dim, action_dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        ep_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss = F.mse_loss(pred, act_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
        if epoch % 40 == 0 or epoch == args.epochs - 1:
            print(f"[bc] epoch {epoch:3d}/{args.epochs}  loss={ep_loss / len(loader):.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden": args.hidden,
        "obs_mode": "compact",        # signal to evaluate_manip to use 12-dim obs
        "hdf5_source": hdf5_path,
        "epochs": args.epochs,
    }, out)
    print(f"[bc] saved {out}")


if __name__ == "__main__":
    main()
