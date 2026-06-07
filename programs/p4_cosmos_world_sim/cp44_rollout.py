"""CP4.4 K-step autoregressive rollout with LoRA fine-tuned Cosmos model.

Runs K-step forward rollout from a real G1 nav episode start frame,
computes SSIM and PSNR vs ground-truth frames.

Usage (run from /tmp/cosmos-predict2/):
    cd /tmp/cosmos-predict2
    python /teamspace/studios/this_studio/Humanoid/programs/p4_cosmos_world_sim/cp44_rollout.py         --lora-ckpt /teamspace/studios/this_studio/Humanoid/checkpoints/p4_cosmos_lora         --data /teamspace/studios/this_studio/Humanoid/datasets/g1_nav         --k 8         --out /teamspace/studios/this_studio/Humanoid/docs/results/cp44_rollout.mp4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_DIR = "/teamspace/studios/this_studio/Humanoid"
COSMOS_ROOT = "/tmp/cosmos-predict2"
BASE_CKPT = f"{REPO_DIR}/checkpoints/cosmos_base/model-480p-4fps.pt"


def load_episode(data_root: str, ep_id: int = 0):
    """Load Bridge-format episode: list of frames, list of actions."""
    import mediapy as mp
    vid_path = os.path.join(data_root, "videos", "train", str(ep_id), "0", "rgb.mp4")
    ann_path = os.path.join(data_root, "annotation", "train", f"{ep_id}.json")

    frames = mp.read_video(vid_path)  # (T, H, W, 3) uint8
    with open(ann_path) as f:
        ann = json.load(f)
    actions = np.array(ann["action"])[:, :6] * 20  # undo Bridge /20 scale back to m/s
    return frames, actions


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute SSIM and PSNR between predicted and ground-truth frames (H,W,3) uint8."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    ssim = structural_similarity(p, g, channel_axis=2, data_range=255)
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    return {"ssim": ssim, "psnr": psnr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-ckpt", default=f"{REPO_DIR}/checkpoints/p4_cosmos_lora")
    parser.add_argument("--data", default=f"{REPO_DIR}/datasets/g1_nav")
    parser.add_argument("--k", type=int, default=8, help="K-step rollout horizon")
    parser.add_argument("--ep", type=int, default=0, help="Episode index to evaluate on")
    parser.add_argument("--out", default=f"{REPO_DIR}/docs/results/cp44_rollout.mp4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if COSMOS_ROOT not in sys.path:
        sys.path.insert(0, COSMOS_ROOT)
    os.chdir(COSMOS_ROOT)

    import torch
    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
    from imaginaire.utils import misc
    from imaginaire.utils.io import save_image_or_video
    import mediapy as mp

    # Load episode
    print(f"Loading episode {args.ep} from {args.data}")
    frames, actions = load_episode(args.data, args.ep)
    T = min(len(frames), len(actions), args.k + 1)

    config = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False
    config.resize_online = False
    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.allow_tf32 = True

    # Load base model + apply LoRA weights if available
    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config,
        dit_path=BASE_CKPT,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=False,
    )

    # Load LoRA adapter if checkpoint exists
    lora_pt = os.path.join(args.lora_ckpt, "lora_adapter.pt")
    if os.path.exists(lora_pt):
        state = torch.load(lora_pt, map_location="cpu")
        pipe.dit.load_state_dict(state, strict=False)
        print(f"LoRA adapter loaded from {lora_pt}")
    else:
        print(f"[WARN] LoRA checkpoint not found at {lora_pt} — using base model")

    print(f"Running {args.k}-step rollout ...")
    cond_frame = frames[0]
    # Bridge actions are already in /20 scale in the JSON; multiply back for LoRA model
    act_seq = actions[:args.k]  # (K, 6) in m/s scale

    pred_video = pipe(cond_frame, act_seq, num_conditional_frames=1,
                      guidance=0, seed=args.seed)
    # pred_video: (T, H, W, 3) uint8

    # Compute metrics frame-by-frame
    print("\nRollout metrics:")
    all_ssim, all_psnr = [], []
    for t in range(min(args.k, len(pred_video), T - 1)):
        pred_f = np.array(pred_video[t])
        gt_f = frames[t + 1]
        # Resize gt to match pred if needed
        if pred_f.shape != gt_f.shape:
            from PIL import Image
            gt_f = np.array(Image.fromarray(gt_f).resize(
                (pred_f.shape[1], pred_f.shape[0]), Image.BILINEAR))
        m = compute_metrics(pred_f, gt_f)
        all_ssim.append(m["ssim"])
        all_psnr.append(m["psnr"])
        print(f"  t={t+1:02d}  SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}dB")

    print(f"\nMean SSIM: {np.mean(all_ssim):.4f}")
    print(f"Mean PSNR: {np.mean(all_psnr):.2f} dB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image_or_video(pred_video, args.out, fps=4)
    print(f"CP4.4 DONE — rollout saved to {args.out}")


if __name__ == "__main__":
    main()
