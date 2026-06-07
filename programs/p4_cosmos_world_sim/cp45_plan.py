"""CP4.5 CEM (Cross-Entropy Method) planning with Cosmos world model.

Samples candidate action sequences, rolls out in world model, ranks by proximity
to goal (uses optical flow magnitude as proxy for forward progress), selects elite
samples, and outputs the best plan + planning visualization video.

Run inside Isaac Lab Docker (needs P3 env for final policy comparison):
    docker exec isaac-lab-base python /workspace/programs/p4_cosmos_world_sim/cp45_plan.py         --cosmos-ckpt /workspace/checkpoints/p4_cosmos_lora         --data /workspace/datasets/g1_nav         --plan-steps 8         --cem-samples 64         --out /workspace/docs/results/cp45_planning.mp4

Or standalone (no Docker required for visualization only):
    cd /tmp/cosmos-predict2
    python /teamspace/studios/this_studio/Humanoid/programs/p4_cosmos_world_sim/cp45_plan.py         --cosmos-ckpt /teamspace/studios/this_studio/Humanoid/checkpoints/p4_cosmos_lora         --data /teamspace/studios/this_studio/Humanoid/datasets/g1_nav         --plan-steps 8 --cem-samples 64         --out /teamspace/studios/this_studio/Humanoid/docs/results/cp45_planning.mp4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_DIR = "/teamspace/studios/this_studio/Humanoid"
COSMOS_ROOT = "/tmp/cosmos-predict2"
BASE_CKPT = f"{REPO_DIR}/checkpoints/cosmos_base/model-480p-4fps.pt"


def score_trajectory(frames: np.ndarray) -> float:
    """Proxy reward: mean optical flow magnitude (forward progress indicator)."""
    import cv2
    total = 0.0
    for t in range(len(frames) - 1):
        f1 = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            f1.astype(np.float32), f2.astype(np.float32),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        total += float(np.mean(np.linalg.norm(flow, axis=2)))
    return total / max(1, len(frames) - 1)


def sample_action_sequences(
    mean: np.ndarray, std: np.ndarray, n: int, H: int
) -> np.ndarray:
    """Sample n action sequences of length H from Gaussian(mean, std).

    Actions are 6D: [vx, vy, 0, 0, 0, omega] in Bridge /20 scale.
    """
    noise = np.random.randn(n, H, 6) * std[None, None, :]
    samples = mean[None, :, :] + noise
    # Zero out dims 2,3,4 (not used for nav)
    samples[:, :, 2:5] = 0.0
    # Clip to reasonable range: ±1.0 in /20 scale = ±20 m/s (much too fast; clip to ±0.15)
    samples = np.clip(samples, -0.15, 0.15)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-ckpt", default=f"{REPO_DIR}/checkpoints/p4_cosmos_lora")
    parser.add_argument("--data", default=f"{REPO_DIR}/datasets/g1_nav")
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--plan-steps", type=int, default=8, help="Planning horizon H")
    parser.add_argument("--cem-samples", type=int, default=64, help="CEM sample count N")
    parser.add_argument("--cem-iters", type=int, default=5, help="CEM refinement iterations")
    parser.add_argument("--elite-frac", type=float, default=0.1, help="Elite fraction for CEM")
    parser.add_argument("--out", default=f"{REPO_DIR}/docs/results/cp45_planning.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if COSMOS_ROOT not in sys.path:
        sys.path.insert(0, COSMOS_ROOT)
    os.chdir(COSMOS_ROOT)

    import torch
    import mediapy as mp
    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
    from imaginaire.utils import misc
    from imaginaire.utils.io import save_image_or_video

    # Load episode start frame
    vid_path = os.path.join(args.data, "videos", "train", str(args.ep), "0", "rgb.mp4")
    frames_gt = mp.read_video(vid_path)
    cond_frame = frames_gt[0]

    config = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False
    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.allow_tf32 = True

    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config,
        dit_path=BASE_CKPT,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=False,
    )

    lora_pt = os.path.join(args.cosmos_ckpt, "lora_adapter.pt")
    if os.path.exists(lora_pt):
        state = torch.load(lora_pt, map_location="cpu")
        pipe.dit.load_state_dict(state, strict=False)
        print(f"LoRA loaded from {lora_pt}")
    else:
        print("[WARN] No LoRA checkpoint — using base model")

    H = args.plan_steps
    N = args.cem_samples
    n_elite = max(1, int(N * args.elite_frac))

    # CEM initial distribution: forward-biased (vx > 0)
    mean = np.zeros((H, 6))
    mean[:, 0] = 0.05   # small forward bias in /20 scale
    std = np.ones(6) * 0.05
    std[2:5] = 0.0       # nav dims unused

    print(f"CEM planning: H={H}, N={N}, {args.cem_iters} iterations ...")
    best_video = None
    best_score = -1e9
    best_actions = None

    for it in range(args.cem_iters):
        candidates = sample_action_sequences(mean, std, N, H)  # (N, H, 6)
        scores = []

        for i, act_seq in enumerate(candidates):
            # Roll out in world model
            video = pipe(cond_frame, act_seq, num_conditional_frames=1,
                         guidance=0, seed=args.seed + i)
            s = score_trajectory(np.array(video))
            scores.append(s)
            if s > best_score:
                best_score = s
                best_video = video
                best_actions = act_seq.copy()

        scores_arr = np.array(scores)
        elite_idx = np.argsort(scores_arr)[-n_elite:]
        elite_seqs = candidates[elite_idx]

        # Update CEM distribution
        mean = elite_seqs.mean(axis=0)
        std_new = elite_seqs.std(axis=0)
        std = np.maximum(std_new, 1e-4)
        std[2:5] = 0.0

        print(f"  Iter {it+1}/{args.cem_iters}: best={best_score:.4f} mean_elite={scores_arr[elite_idx].mean():.4f}")

    print(f"\nBest plan score: {best_score:.4f}")
    print("Best action sequence (vx, vy, omega in /20 scale):")
    for t, a in enumerate(best_actions):
        print(f"  t={t}: vx={a[0]:.4f} vy={a[1]:.4f} omega={a[5]:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image_or_video(best_video, args.out, fps=4)
    print(f"CP4.5 DONE — planning video saved to {args.out}")


if __name__ == "__main__":
    main()
