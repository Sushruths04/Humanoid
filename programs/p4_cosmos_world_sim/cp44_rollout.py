"""CP4.4: K-step action-conditioned rollout evaluation.

Uses Bridge dataset (same as cp43_train.py). Loads trained LoRA, generates
frames via Euler ODE reverse diffusion, reports SSIM/PSNR vs real frames.

DoD: generation runs; SSIM/PSNR reported; side-by-side video saved.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = "/teamspace/studios/this_studio/Humanoid"
COSMOS_ROOT = "/tmp/cosmos-predict2"
BASE_CKPT = f"{REPO}/checkpoints/cosmos_base/model-480p-4fps.pt"
DATA = f"{REPO}/datasets/g1_nav"
LORA_CKPT = f"{REPO}/checkpoints/p4_cosmos_lora/lora_adapter.pt"


def load_bridge_episode(data_root: str, ep_id: int):
    import mediapy as mp
    import numpy as np
    vid_path = os.path.join(data_root, "videos", "train", str(ep_id), "0", "rgb.mp4")
    ann_path = os.path.join(data_root, "annotation", "train", f"{ep_id}.json")
    frames = mp.read_video(vid_path)
    with open(ann_path) as f:
        ann = json.load(f)
    actions = (np.array(ann["action"])[:, :6] * 20)[:len(frames)]
    gripper = np.array(ann["continuous_gripper_state"])[1:len(frames)+1, None]
    return frames, np.concatenate([actions, gripper[:len(frames)]], axis=1)


def prepare_inputs(frames, actions):
    """Pad frames so (T-1)%4==0. Returns vid [1,3,T,H,W], act [1,12,7], gt numpy [T,H,W,3]."""
    import torch
    import numpy as np
    T = min(len(frames), 12)
    frames, actions = frames[:T], actions[:T]
    pad = (4 - (T - 1) % 4) % 4
    if pad > 0:
        frames = np.concatenate([frames, np.tile(frames[-1:], (pad, 1, 1, 1))], axis=0)
    vid = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
    vid = (vid / 255.0) * 2.0 - 1.0
    vid = vid.to(torch.bfloat16)
    act = torch.from_numpy(actions[:12]).unsqueeze(0).to("cuda", torch.bfloat16)
    return vid, act, frames  # frames is uint8 numpy [T_pad, H, W, 3]


def generate_frames(pipe, vid_t, act_t, n_steps: int = 20):
    """
    Euler ODE reverse diffusion (t: 0.99 → 0.01).
    vid_t: [1, 3, T_pad, H, W] bfloat16 in [-1, 1]
    act_t: [1, 12, 7] bfloat16
    Returns: uint8 numpy [T, H, W, 3]
    """
    import torch
    import numpy as np
    from cosmos_predict2.conditioner import ActionCondition, DataType

    with torch.no_grad():
        latent = pipe.tokenizer.encode(vid_t)  # [1, C, T_lat, H_lat, W_lat]

    B, C, T_lat, H_lat, W_lat = latent.shape
    H_vid, W_vid = vid_t.shape[-2], vid_t.shape[-1]

    mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=latent.device, dtype=latent.dtype)
    mask[:, :, :1] = 1.0  # condition on first latent frame only

    condition = ActionCondition(
        crossattn_emb=torch.zeros(B, 256, 1024, device=latent.device, dtype=pipe.precision),
        data_type=DataType.VIDEO,
        padding_mask=torch.zeros(B, 1, H_vid, W_vid, device=latent.device, dtype=pipe.precision),
        fps=None,
        use_video_condition=True,
        gt_frames=latent,
        condition_video_input_mask_B_C_T_H_W=mask,
        action=act_t,
    )

    x_t = torch.randn_like(latent)
    ts = torch.linspace(0.99, 0.01, n_steps + 1, device=latent.device, dtype=pipe.precision)

    with torch.no_grad():
        for i in range(n_steps):
            t_curr = float(ts[i])
            t_next = float(ts[i + 1])
            sigma = t_curr / (1.0 - t_curr + 1e-8)
            sigma_t = torch.full((B,), sigma, device=x_t.device, dtype=pipe.precision)
            pred = pipe.denoise(x_t, sigma_t, condition)
            # RF velocity: v = (x_t - x0) / t; Euler step
            velocity = (x_t - pred.x0) / (t_curr + 1e-8)
            x_t = x_t + (t_next - t_curr) * velocity

        decoded = pipe.tokenizer.decode(x_t)  # [1, 3, T, H, W] in [-1, 1]

    frames = decoded[0].permute(1, 2, 3, 0).cpu().float().numpy()
    return np.clip((frames + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--checkpoint", default=LORA_CKPT)
    parser.add_argument("--model-dir", default=os.path.dirname(BASE_CKPT))
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=20)
    parser.add_argument("--out", default=f"{REPO}/docs/results/cp44_rollout.mp4")
    args = parser.parse_args()

    if COSMOS_ROOT not in sys.path:
        sys.path.insert(0, COSMOS_ROOT)
    os.chdir(COSMOS_ROOT)

    import torch
    import numpy as np
    import imageio
    from peft import LoraConfig, inject_adapter_in_model
    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
    from imaginaire.utils import misc

    misc.set_random_seed(seed=0, by_rank=True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    config = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False
    config.resize_online = False

    base_ckpt = os.path.join(args.model_dir, "model-480p-4fps.pt")
    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config, dit_path=base_ckpt,
        use_text_encoder=False, device="cuda",
        torch_dtype=torch.bfloat16, load_prompt_refiner=False,
    )
    print(f"Base model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    cfg = LoraConfig(
        r=16, lora_alpha=16, init_lora_weights=True,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"],
    )
    pipe.dit = inject_adapter_in_model(cfg, pipe.dit)
    lora_state = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    missing, unexpected = pipe.dit.load_state_dict(lora_state, strict=False)
    print(f"LoRA loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    pipe.dit.eval()

    ann_dir = Path(args.data) / "annotation" / "train"
    ep_ids = sorted([int(p.stem) for p in ann_dir.glob("*.json")])[:args.num_episodes]
    print(f"Evaluating {len(ep_ids)} episodes, {args.denoise_steps} denoise steps each ...")

    all_ssim, all_psnr, all_mse = [], [], []
    video_frames = []

    for ep_id in ep_ids:
        frames, actions = load_bridge_episode(args.data, ep_id)
        vid_t, act_t, gt = prepare_inputs(frames, actions)
        print(f"  Episode {ep_id}: T_pad={vid_t.shape[2]} ...")

        gen = generate_frames(pipe, vid_t, act_t, n_steps=args.denoise_steps)
        # gen: [T_gen, H_gen, W_gen, 3] uint8

        # Resize gt to match Cosmos output resolution if needed
        H, W = gen.shape[1], gen.shape[2]
        if gt.shape[1] != H or gt.shape[2] != W:
            import cv2
            gt = np.stack([cv2.resize(f.astype(np.uint8), (W, H)) for f in gt])

        # Metrics on future frames only (skip frame 0 = conditioning)
        T_eval = min(gen.shape[0] - 1, gt.shape[0] - 1)
        p_eval = gen[1:T_eval+1].astype(np.float32)
        r_eval = gt[1:T_eval+1].astype(np.float32)

        mse_val = float(np.mean((p_eval - r_eval) ** 2))
        psnr_val = float(20 * np.log10(255.0 / (np.sqrt(mse_val) + 1e-8)))
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            ssim_val = float(np.mean([
                ssim_fn(p.astype(np.uint8), r.astype(np.uint8), channel_axis=-1, data_range=255)
                for p, r in zip(p_eval, r_eval)
            ]))
        except ImportError:
            ssim_val = float("nan")

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_mse.append(mse_val)
        print(f"  → SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  MSE={mse_val:.1f}")

        N = min(12, gt.shape[0], gen.shape[0])
        for fi in range(N):
            video_frames.append(np.concatenate([gt[fi], gen[fi]], axis=1))

    print(f"\n===== CP4.4 Results =====")
    print(f"Episodes: {len(ep_ids)}")
    print(f"SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}  (1.0=perfect)")
    print(f"PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"MSE:  {np.mean(all_mse):.1f} ± {np.std(all_mse):.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, video_frames, fps=4, quality=8)
    print(f"\nRollout video (real left / generated right): {args.out}")
    print("CP4.4 DONE")


if __name__ == "__main__":
    main()
