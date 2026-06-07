"""CP4.3: LoRA post-train Cosmos Predict 2.5 on G1 nav (frame_t, action_t) → frame_t+1.

SMOKE GATE: always run with --smoke --max-steps 2 before the full run.

Usage:
    # Smoke test (2 steps, cheap):
    python -m programs.p4_cosmos_world_sim.cp43_train \
        --data datasets/g1_nav_cosmos.h5 \
        --model-dir checkpoints/cosmos_base/ \
        --smoke --max-steps 2 --out /tmp/cosmos_smoke/

    # Full training:
    python -m programs.p4_cosmos_world_sim.cp43_train \
        --data datasets/g1_nav_cosmos.h5 \
        --model-dir checkpoints/cosmos_base/ \
        --lora-rank 16 --max-steps 5000 \
        --save-every 500 --out checkpoints/p4_cosmos_lora/

    # Eval-only diff-actions test (after training):
    python -m programs.p4_cosmos_world_sim.cp43_train \
        --eval-only \
        --checkpoint checkpoints/p4_cosmos_lora/ \
        --out docs/results/cp43_action_diff.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _load_cosmos_for_training(model_dir: str, lora_rank: int, device: str):
    """Load Cosmos model and attach LoRA adapters via PEFT."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    # Load base model — exact API depends on cosmos-predict2 version.
    # Check /tmp/cosmos-predict2/README.md § Training / Fine-tuning.
    try:
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # type: ignore
        base_model = Video2WorldPipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16
        )
        transformer = base_model.transformer  # the diffusion transformer
    except (ImportError, AttributeError):
        # Fallback: try direct model load
        from cosmos_predict2 import CosmosPredictModel  # type: ignore
        transformer = CosmosPredictModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        base_model = None

    # LoRA config — target the attention projection layers
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer = transformer.to(device)

    return base_model, transformer


def _make_dataloader(h5_path: str, batch_size: int, smoke: bool):
    """Yield (frame_t, action_t, frame_t+1) batches from HDF5."""
    import h5py
    import numpy as np
    import torch

    with h5py.File(h5_path, "r") as f:
        n = f["frame_t"].shape[0]
        frame_t = f["frame_t"][:]    # (N, 64, 64, 3)
        action_t = f["action_t"][:]  # (N, 29)
        frame_t1 = f["frame_t1"][:] # (N, 64, 64, 3)

    if smoke:
        n = min(n, batch_size * 4)
        frame_t, action_t, frame_t1 = frame_t[:n], action_t[:n], frame_t1[:n]

    # Normalise frames to [-1, 1] (standard for diffusion models)
    ft = (frame_t.astype(np.float32) / 127.5) - 1.0   # (N, 64, 64, 3)
    ft1 = (frame_t1.astype(np.float32) / 127.5) - 1.0

    # Rearrange to (N, 3, H, W) for PyTorch convention
    ft = np.transpose(ft, (0, 3, 1, 2))
    ft1 = np.transpose(ft1, (0, 3, 1, 2))

    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n - batch_size + 1, batch_size):
            b = indices[start: start + batch_size]
            yield (
                torch.from_numpy(ft[b]),
                torch.from_numpy(action_t[b]),
                torch.from_numpy(ft1[b]),
            )


def _diff_actions_eval(base_model, ckpt_dir: str, out_path: str, device: str) -> None:
    """Generate two videos with different actions and save side-by-side."""
    import imageio
    import numpy as np
    import torch

    print("Running diff-actions eval ...")

    # Load fine-tuned pipeline
    try:
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # type: ignore
        pipe = Video2WorldPipeline.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)
    except ImportError:
        raise ImportError(
            "Cannot load Cosmos pipeline for eval. "
            "Check cosmos-predict2 docs for the pipeline class name."
        )

    # Use a fixed initial frame (zeros = gray frame; replace with a real frame if desired)
    init_frame = np.full((64, 64, 3), 128, dtype=np.uint8)

    # Two contrasting actions: full forward vs full backward (joint 0 = hip pitch)
    action_forward = np.zeros(29, dtype=np.float32)
    action_forward[0] = 1.0   # forward command

    action_backward = np.zeros(29, dtype=np.float32)
    action_backward[0] = -1.0  # backward command

    videos = []
    for label, action in [("forward", action_forward), ("backward", action_backward)]:
        action_t = torch.tensor(action, dtype=torch.bfloat16).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pipe(image=init_frame, action=action_t, num_frames=8)
        frames = output.frames if hasattr(output, "frames") else output[0]
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy()
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
        videos.append(frames)
        print(f"  {label}: generated {len(frames)} frames")

    # Stack side by side: (T, H, W*2, 3)
    side_by_side = np.concatenate([videos[0], videos[1]], axis=2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimwrite(out_path, side_by_side, fps=8, quality=8)
    print(f"Diff-actions video saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CP4.3: LoRA post-train Cosmos Predict")
    parser.add_argument("--data", help="Path to g1_nav_cosmos.h5")
    parser.add_argument("--model-dir", help="Base Cosmos model weights directory")
    parser.add_argument("--checkpoint", help="Fine-tuned checkpoint dir (for --eval-only)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true", help="Smoke test: tiny data, 2 steps")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run diff-actions eval")
    parser.add_argument("--out", required=True, help="Output dir (train) or mp4 path (eval-only)")
    args = parser.parse_args()

    import torch
    import bitsandbytes as bnb  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"Device: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb} GB")

    if args.eval_only:
        if not args.checkpoint:
            print("--checkpoint required for --eval-only")
            sys.exit(1)
        _diff_actions_eval(None, args.checkpoint, args.out, device)
        return

    if not args.data or not args.model_dir:
        print("--data and --model-dir required for training")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        print("=== SMOKE TEST (2 steps) ===")
        args.max_steps = 2

    _, transformer = _load_cosmos_for_training(args.model_dir, args.lora_rank, device)

    # 8-bit Adam — reduces VRAM by ~4x vs fp32 Adam
    optimizer = bnb.optim.AdamW8bit(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    loader = _make_dataloader(args.data, args.batch_size, smoke=args.smoke)

    # Gradient checkpointing to reduce VRAM further
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    transformer.train()

    print(f"Starting training: {args.max_steps} steps, lr={args.lr}, lora_rank={args.lora_rank}")
    start = time.time()

    for step, (ft, at, ft1) in enumerate(loader):
        if step >= args.max_steps:
            break

        ft = ft.to(device, dtype=torch.bfloat16)
        at = at.to(device, dtype=torch.bfloat16)
        ft1 = ft1.to(device, dtype=torch.bfloat16)

        # Forward pass — the exact API call depends on cosmos-predict2 internals.
        # Cosmos uses a diffusion loss; the transformer predicts noise given (ft, at).
        # On machine, check /tmp/cosmos-predict2/training/train.py for the exact call.
        # The pattern is typically:
        #   noise = torch.randn_like(ft1)
        #   t = torch.randint(0, T, (B,), device=device)
        #   x_noisy = q_sample(ft1, t, noise)
        #   pred = transformer(x_noisy, t, context=ft, action=at)
        #   loss = F.mse_loss(pred, noise)
        # We implement a minimal version here; update from cookbook if needed.
        try:
            loss = transformer(frame_t=ft, action=at, frame_t1=ft1, return_loss=True)
        except TypeError:
            # Fallback: assume a simpler callable interface
            loss = transformer(ft, at, ft1)

        if hasattr(loss, "loss"):
            loss = loss.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()

        elapsed = time.time() - start
        print(f"Step {step+1}/{args.max_steps}  loss={loss.item():.4f}  ({elapsed:.1f}s)")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = os.path.join(args.out, f"checkpoint_{step+1}")
            transformer.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if args.smoke:
        if device == "cuda":
            vram_used = torch.cuda.max_memory_allocated() // 1024**3
            print(f"\nSmoke test PASSED — VRAM used: {vram_used} GB / {vram_gb} GB")
        print("Smoke test PASSED — proceed to full training run.")
    else:
        print(f"\nCP4.3 DONE — {args.max_steps} steps complete. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
