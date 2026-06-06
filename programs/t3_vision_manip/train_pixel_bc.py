"""T3 — Train pixel-conditioned BC policy on LIBERO Spatial HDF5 demos.

Loads agentview_image + actions from all 10 libero_spatial tasks, trains a
ResNet18+MLP policy via MSE BC, saves checkpoint.

Usage:
    PYTHONUNBUFFERED=1 python -m programs.t3_vision_manip.train_pixel_bc \
        --data-dir /teamspace/studios/this_studio/libero_datasets/libero_spatial \
        --out programs/checkpoints/t3_pixel_bc/pixel_bc.pt \
        --epochs 200 --batch-size 64
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
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True,
                   help="Dir containing libero_spatial .hdf5 demo files")
    p.add_argument("--out", type=str,
                   default="programs/checkpoints/t3_pixel_bc/pixel_bc.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-demos", type=int, default=50,
                   help="Max demos per task file (50 = use all)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--result-doc", type=str, default="docs/results/t3_pixel_bc.md")
    return p.parse_args()


class LiberoPixelDataset(Dataset):
    """Loads (agentview_image, action) pairs from LIBERO HDF5 files.

    Pre-resizes and normalizes all images at init time so __getitem__ is a
    cheap tensor slice — no per-sample CPU transform during training.
    """

    def __init__(self, hdf5_paths: list[str], max_demos: int, img_size: int):
        import h5py
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])

        imgs_raw, acts = [], []
        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                demo_keys = sorted(f["data"].keys())[:max_demos]
                for dk in demo_keys:
                    demo = f["data"][dk]
                    img_seq = np.array(demo["obs"]["agentview_rgb"])    # (T, H, W, 3) uint8
                    act_seq = np.array(demo["actions"])                 # (T, 7)
                    imgs_raw.append(img_seq[:-1])
                    acts.append(act_seq[:-1])
            print(f"  loaded {Path(path).name}: {len(demo_keys)} demos")

        imgs_np = np.concatenate(imgs_raw, axis=0)    # (N, H, W, 3) uint8
        self.acts = torch.from_numpy(
            np.concatenate(acts, axis=0).astype(np.float32))
        print(f"[dataset] {len(imgs_np)} transitions — pre-processing images...")

        # Pre-normalize once: (N, H, W, 3) uint8 → (N, 3, H, W) float32
        # Keep native 128×128 resolution — ResNet18 global avg-pool handles any size.
        t = torch.from_numpy(imgs_np).float() / 255.0   # (N, H, W, 3)
        t = t.permute(0, 3, 1, 2).contiguous()            # (N, 3, H, W) contiguous
        t = (t - mean[None, :, None, None]) / std[None, :, None, None]
        # Move entire dataset to GPU VRAM — eliminates CPU→GPU transfer every batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imgs = t.to(device)
        self.acts = self.acts.to(device)
        print(f"[dataset] pre-processing done — imgs on {device}: {self.imgs.shape}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.acts[idx]


def main():
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hdf5_paths = sorted(Path(args.data_dir).glob("*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 files in {args.data_dir}")
    print(f"[t3] found {len(hdf5_paths)} task files in {args.data_dir}")

    dataset = LiberoPixelDataset(
        [str(p) for p in hdf5_paths],
        max_demos=args.max_demos,
        img_size=args.img_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    from programs.t3_vision_manip.pixel_bc_policy import PixelBCPolicy
    model = PixelBCPolicy(action_dim=7, freeze_encoder=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"[t3] training {args.epochs} epochs  batch={args.batch_size}  device={device}")
    init_loss = final_loss = 0.0

    for epoch in range(args.epochs):
        ep_loss = 0.0
        for imgs, acts in loader:
            # data already on GPU (moved at dataset init)
            pred = model(imgs)
            loss = F.mse_loss(pred, acts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
        sched.step()
        avg = ep_loss / len(loader)
        if epoch == 0:
            init_loss = avg
        final_loss = avg
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[t3] epoch {epoch:3d}/{args.epochs}  loss={avg:.5f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "action_dim": 7,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "num_tasks": len(hdf5_paths),
        "num_transitions": len(dataset),
    }, out)
    print(f"[t3] checkpoint saved → {out}")
    print(f"[t3] loss: {init_loss:.5f} → {final_loss:.5f}")

    doc = Path(args.result_doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join([
        "# T3 Pixel BC — Training Results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Tasks | {len(hdf5_paths)} (all libero_spatial) |",
        f"| Transitions | {len(dataset)} |",
        f"| Epochs | {args.epochs} |",
        f"| Batch size | {args.batch_size} |",
        f"| Initial loss | {init_loss:.5f} |",
        f"| Final loss | **{final_loss:.5f}** |",
        f"| Checkpoint | `{out}` |",
        "",
        "_Eval results appended by evaluate_pixel_bc.py_",
    ]))
    print(f"[t3] doc → {doc}")


if __name__ == "__main__":
    main()
