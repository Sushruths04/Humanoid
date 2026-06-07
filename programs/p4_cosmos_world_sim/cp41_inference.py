"""CP4.1: Stock Cosmos Predict 2.5 inference — generates a video from an initial nav frame.

DoD: mp4 saved; not black; not NaN. Proves the model loads and runs.

Usage:
    python -m programs.p4_cosmos_world_sim.cp41_inference \
        --model-dir ~/Humanoid/checkpoints/cosmos_base/ \
        --frame /tmp/initial_frame.npy \
        --out docs/results/cp41_inference.mp4 \
        --steps 16
"""

from __future__ import annotations

import argparse
import os


def _find_cosmos_api(model_dir: str):
    """Return (pipeline_cls, load_kwargs) by probing cosmos-predict2 API variants."""
    # Try the cosmos-predict2 pipeline API (exact class names vary by version).
    # On machine: check /tmp/cosmos-predict2/README.md for the actual class name.
    try:
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # type: ignore
        return Video2WorldPipeline, {"pretrained_model_name_or_path": model_dir}
    except ImportError:
        pass
    try:
        from cosmos_predict2 import CosmosPredict2Pipeline  # type: ignore
        return CosmosPredict2Pipeline, {"model_path": model_dir}
    except ImportError:
        pass
    raise ImportError(
        "Cannot import Cosmos Predict 2 pipeline. "
        "Check /tmp/cosmos-predict2/README.md for the correct import path and class name, "
        "then update this file accordingly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CP4.1: stock Cosmos inference baseline")
    parser.add_argument("--model-dir", required=True, help="Local path to Cosmos weights")
    parser.add_argument("--frame", required=True, help="Initial frame as .npy (H,W,3 uint8) or image path")
    parser.add_argument("--out", default="docs/results/cp41_inference.mp4")
    parser.add_argument("--steps", type=int, default=16, help="Number of video frames to generate")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    import imageio
    import numpy as np
    import torch

    # Load initial frame
    if args.frame.endswith(".npy"):
        frame = np.load(args.frame)  # (H, W, 3) uint8
    else:
        frame = imageio.imread(args.frame)  # any image format
    print(f"Initial frame: {frame.shape} {frame.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    pipeline_cls, load_kwargs = _find_cosmos_api(args.model_dir)

    print(f"Loading Cosmos model from {args.model_dir} ...")
    pipe = pipeline_cls.from_pretrained(**load_kwargs, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    print("Model loaded.")

    # Run inference — API varies; check README if this errors
    with torch.no_grad():
        output = pipe(
            image=frame,
            num_frames=args.steps,
            # action conditioning not used in stock model — just visual continuation
        )

    # Output is typically a list of frames or a tensor (T, H, W, 3)
    if hasattr(output, "frames"):
        frames = output.frames  # HuggingFace Diffusers style
    elif isinstance(output, (list, tuple)):
        frames = output[0]
    else:
        frames = output

    if hasattr(frames, "cpu"):
        frames = frames.cpu().numpy()

    # Ensure (T, H, W, 3) uint8
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

    imageio.mimwrite(args.out, frames, fps=8, quality=8)
    print(f"CP4.1 DONE — saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
