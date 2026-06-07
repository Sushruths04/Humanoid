"""CP4.4: K-step action-conditioned rollout — imagined vs real frames, fidelity report.

DoD: K-step rollout runs; SSIM + PSNR vs real frames reported; rollout video saved.

Usage:
    python -m programs.p4_cosmos_world_sim.cp44_rollout \
        --checkpoint checkpoints/p4_cosmos_lora/ \
        --data datasets/g1_nav_cosmos.h5 \
        --k-steps 8 \
        --out docs/results/cp44_rollout.mp4
"""

from __future__ import annotations

import argparse
import os

import h5py
import imageio
import numpy as np
import torch


def ssim_batch(pred: np.ndarray, real: np.ndarray) -> float:
    """Compute mean SSIM between two (N, H, W, 3) uint8 arrays."""
    from skimage.metrics import structural_similarity as ssim  # type: ignore

    scores = []
    for p, r in zip(pred, real):
        s = ssim(p, r, channel_axis=-1, data_range=255)
        scores.append(s)
    return float(np.mean(scores))


def psnr_batch(pred: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def main() -> None:
    parser = argparse.ArgumentParser(description="CP4.4: K-step Cosmos rollout evaluation")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned Cosmos checkpoint dir")
    parser.add_argument("--data", required=True, help="Path to g1_nav_cosmos.h5")
    parser.add_argument("--k-steps", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=16, help="How many rollouts to evaluate")
    parser.add_argument("--out", default="docs/results/cp44_rollout.mp4")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned Cosmos pipeline
    try:
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # type: ignore
        pipe = Video2WorldPipeline.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16
        ).to(device)
    except ImportError as e:
        raise ImportError(
            "Cannot import Cosmos pipeline. Check cosmos-predict2 docs for correct import."
        ) from e

    pipe.eval()

    # Load K-step sequences from dataset
    with h5py.File(args.data, "r") as f:
        n = f["frame_t"].shape[0]
        # Pick random start points with room for k steps
        starts = np.random.choice(n - args.k_steps, args.num_sequences, replace=False)
        ft_real = []
        at_seq = []
        for s in starts:
            ft_real.append(f["frame_t"][s: s + args.k_steps])   # (k, 64, 64, 3)
            at_seq.append(f["action_t"][s: s + args.k_steps])   # (k, 29)

    ft_real = np.array(ft_real)   # (num_seq, k, 64, 64, 3)
    at_seq = np.array(at_seq)     # (num_seq, k, 29)

    imagined_frames = []  # (num_seq, k, 64, 64, 3) uint8

    print(f"Running {args.num_sequences} rollouts of {args.k_steps} steps ...")
    for i in range(args.num_sequences):
        seq_frames = []
        current_frame = ft_real[i, 0]  # start from real first frame

        for k in range(args.k_steps):
            action = torch.tensor(at_seq[i, k], dtype=torch.bfloat16).unsqueeze(0).to(device)
            with torch.no_grad():
                output = pipe(image=current_frame, action=action, num_frames=1)

            frame = output.frames if hasattr(output, "frames") else output[0]
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]  # (H, W, 3) or (1, H, W, 3) → (H, W, 3)
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

            seq_frames.append(frame)
            current_frame = frame  # auto-regressive: use predicted frame as next input

        imagined_frames.append(np.array(seq_frames))  # (k, H, W, 3)

        if (i + 1) % 4 == 0:
            print(f"  Sequence {i+1}/{args.num_sequences}")

    imagined_frames = np.array(imagined_frames)  # (num_seq, k, H, W, 3)

    # Fidelity metrics (flatten seq and step dims for batch computation)
    pred_flat = imagined_frames.reshape(-1, *imagined_frames.shape[2:])
    real_flat = ft_real.reshape(-1, *ft_real.shape[2:])

    try:
        from skimage.metrics import structural_similarity  # noqa: F401
        ssim_val = ssim_batch(pred_flat, real_flat)
        psnr_val = psnr_batch(pred_flat, real_flat)
        mse_val = float(np.mean((pred_flat.astype(np.float32) - real_flat.astype(np.float32)) ** 2))
    except ImportError:
        ssim_val = float("nan")
        psnr_val = float("nan")
        mse_val = float(np.mean((pred_flat.astype(np.float32) - real_flat.astype(np.float32)) ** 2))
        print("  [note] scikit-image not found; SSIM/PSNR skipped. Install with: pip install scikit-image")

    print(f"\nK-step rollout fidelity (vs real frames, K={args.k_steps}):")
    print(f"  SSIM: {ssim_val:.4f}  (1.0 = perfect)")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  MSE:  {mse_val:.3f}")

    # Save side-by-side video: imagined (top) vs real (bottom)
    # Use first sequence for the video
    real_seq = ft_real[0]           # (k, H, W, 3)
    imag_seq = imagined_frames[0]   # (k, H, W, 3)

    # Pad real to same H/W as imagined in case of resize
    if real_seq.shape != imag_seq.shape:
        import cv2  # type: ignore
        real_seq = np.stack([
            cv2.resize(f, (imag_seq.shape[2], imag_seq.shape[1])) for f in real_seq
        ])

    combined = np.concatenate([real_seq, imag_seq], axis=1)   # (k, H*2, W, 3)
    imageio.mimwrite(args.out, combined, fps=4, quality=8)

    print(f"\nRollout video (real top / imagined bottom) saved: {args.out}")
    print("CP4.4 DONE")


if __name__ == "__main__":
    main()
