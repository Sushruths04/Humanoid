"""CP4.1: Stock Cosmos Predict2 action-conditioned inference on a G1 nav frame.

Uses the real cosmos-predict2 API: Video2WorldActionConditionedPipeline.
Run from /tmp/cosmos-predict2/ directory (needs megatron/imaginaire in path).

Usage (run from /tmp/cosmos-predict2/):
    cd /tmp/cosmos-predict2
    export HF_TOKEN=hf_...
    python /teamspace/studios/this_studio/Humanoid/programs/p4_cosmos_world_sim/cp41_inference.py         --video /teamspace/studios/this_studio/Humanoid/videos/p3_vision_nav/p3_vision_nav_model499.mp4         --out /teamspace/studios/this_studio/Humanoid/docs/results/cp41_inference.mp4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_dummy_annotation(num_frames: int, out_path: str) -> None:
    """Create a Bridge-format JSON with zero velocity commands (robot standing still)."""
    annotation = {
        "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * num_frames,
        "continuous_gripper_state": [0.0] * (num_frames + 1),
        "state": [[0.0] * 6] * num_frames,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotation, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input nav video (.mp4)")
    parser.add_argument("--out", default="docs/results/cp41_inference.mp4")
    parser.add_argument("--chunk-size", type=int, default=12, help="Frames to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Must run from cosmos-predict2 root (megatron needs this)
    cosmos_root = "/tmp/cosmos-predict2"
    if os.path.exists(cosmos_root) and cosmos_root not in sys.path:
        os.chdir(cosmos_root)

    import torch
    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
    from imaginaire.utils import misc
    from imaginaire.utils.io import save_image_or_video
    import mediapy as mp

    print(f"Loading Cosmos action-conditioned pipeline (2B, 480p, 4fps) ...")
    config = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    dit_path = "/teamspace/studios/this_studio/Humanoid/checkpoints/cosmos_base/model-480p-4fps.pt"
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config,
        dit_path=dit_path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=False,
    )
    print("Model loaded.")

    # Read first frame from the P3 nav video
    video_frames = mp.read_video(args.video)  # (T, H, W, 3)
    first_frame = video_frames[0]             # (H, W, 3) uint8
    print(f"Input frame: {first_frame.shape}")

    # Create dummy annotation (zero velocity — baseline test)
    ann_path = "/tmp/cp41_annotation.json"
    make_dummy_annotation(args.chunk_size, ann_path)
    with open(ann_path) as f:
        data = json.load(f)
    action_ee = np.array(data["action"])[:, :6] * 20
    gripper = np.array(data["continuous_gripper_state"])[1:args.chunk_size+1, None]
    actions = np.concatenate([action_ee[:args.chunk_size], gripper[:args.chunk_size]], axis=1)

    print(f"Running inference for {args.chunk_size} frames ...")
    video = pipe(first_frame, actions[:args.chunk_size], num_conditional_frames=1,
                 guidance=0, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image_or_video(video, args.out, fps=4)
    print(f"CP4.1 DONE — saved to {args.out}")


if __name__ == "__main__":
    main()
