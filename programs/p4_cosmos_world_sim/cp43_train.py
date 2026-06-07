"""CP4.3 LoRA fine-tuning of Cosmos Predict2 2B on G1 nav Bridge data.

Custom PEFT training loop using pipe.denoise() with ActionCondition.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = "/teamspace/studios/this_studio/Humanoid"
COSMOS_ROOT = "/tmp/cosmos-predict2"
BASE_CKPT = f"{REPO}/checkpoints/cosmos_base/model-480p-4fps.pt"
DATA = f"{REPO}/datasets/g1_nav"


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
    full_actions = np.concatenate([actions, gripper[:len(frames)]], axis=1)
    return frames, full_actions


def add_lora(model, rank: int = 16):
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights=True,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"],
    )
    model = inject_adapter_in_model(cfg, model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def compute_flow_loss(pipe, video_t, actions_t):
    """
    video_t: [B, 3, T_padded, H, W] where (T_padded-1)%4 == 0
    actions_t: [B, 12, 7] — exactly 12 frames to match action_dim=84
    """
    import torch
    from cosmos_predict2.conditioner import ActionCondition, DataType

    with torch.no_grad():
        latent = pipe.tokenizer.encode(video_t)  # [B, C, T_lat, H_lat, W_lat]

    B, C, T_lat, H_lat, W_lat = latent.shape
    H_vid, W_vid = video_t.shape[-2], video_t.shape[-1]

    # Rectified flow noise schedule: t ~ U[0,1], sigma = t/(1-t)
    t_rf = torch.rand(B, device=latent.device, dtype=pipe.precision)
    sigma = t_rf / (1.0 - t_rf + 1e-8)

    epsilon = torch.randn_like(latent)
    t_bc = t_rf.view(B, 1, 1, 1, 1).to(latent.dtype)
    xt = (1.0 - t_bc) * latent + t_bc * epsilon

    # Conditioning mask: first latent frame is the conditioning frame
    mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=latent.device, dtype=latent.dtype)
    mask[:, :, :1] = 1.0

    condition = ActionCondition(
        crossattn_emb=torch.zeros(B, 256, 1024, device=latent.device, dtype=pipe.precision),
        data_type=DataType.VIDEO,
        padding_mask=torch.zeros(B, 1, H_vid, W_vid, device=latent.device, dtype=pipe.precision),
        fps=None,
        use_video_condition=True,
        gt_frames=latent,
        condition_video_input_mask_B_C_T_H_W=mask,
        action=actions_t,
    )

    pred = pipe.denoise(xt, sigma, condition)
    loss = ((pred.x0 - latent) ** 2).mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=f"{REPO}/checkpoints/p4_cosmos_lora")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = 2
        args.save_every = 2
        print("=== SMOKE TEST (2 steps) ===")

    if COSMOS_ROOT not in sys.path:
        sys.path.insert(0, COSMOS_ROOT)
    os.chdir(COSMOS_ROOT)

    import torch
    import numpy as np
    from imaginaire.utils import misc
    from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
    from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline

    misc.set_random_seed(seed=42, by_rank=True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    config = get_cosmos_predict2_action_conditioned_pipeline(model_size="2B", resolution="480", fps=4)
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False
    config.resize_online = False

    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=config, dit_path=BASE_CKPT,
        use_text_encoder=False, device="cuda",
        torch_dtype=torch.bfloat16, load_prompt_refiner=False,
    )
    print(f"Base model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pipe.dit.train()
    pipe.dit = add_lora(pipe.dit, rank=args.lora_rank)
    lora_params = [p for p in pipe.dit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    ann_dir = Path(args.data) / "annotation" / "train"
    ep_ids = sorted([int(p.stem) for p in ann_dir.glob("*.json")])
    print(f"Training: {args.max_steps} steps over {len(ep_ids)} episodes")

    os.makedirs(args.out, exist_ok=True)
    step = 0
    losses = []

    while step < args.max_steps:
        for ep_id in ep_ids:
            if step >= args.max_steps:
                break
            frames, actions = load_bridge_episode(args.data, ep_id)
            T = min(len(frames), 12)
            frames, actions = frames[:T], actions[:T]

            # Cosmos VAE encodes frame-0 alone then chunks of 4.
            # Remainder (T-1)%4 != 0 causes T=2<kernel=3 at the deepest temporal
            # downsampling stage. Pad so (T-1) is divisible by 4.
            pad = (4 - (T - 1) % 4) % 4
            if pad > 0:
                frames = np.concatenate([frames, np.tile(frames[-1:], (pad, 1, 1, 1))], axis=0)

            vid = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
            vid = (vid / 255.0) * 2.0 - 1.0
            vid = vid.to(torch.bfloat16)
            # Actions: exactly 12 frames to match action_dim=84 (12*7)
            act = torch.from_numpy(actions[:12]).unsqueeze(0).to("cuda", torch.bfloat16)

            optimizer.zero_grad()
            loss = compute_flow_loss(pipe, vid, act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1

            if step % 10 == 0 or step <= 3:
                print(f"Step {step}/{args.max_steps} | loss={loss.item():.4f} | gpu={torch.cuda.memory_allocated()/1e9:.1f}GB")

            if step % args.save_every == 0 or step == args.max_steps:
                ckpt = {k: v for k, v in pipe.dit.state_dict().items() if "lora" in k.lower()}
                p = os.path.join(args.out, f"lora_step{step:04d}.pt")
                torch.save(ckpt, p)
                torch.save(ckpt, os.path.join(args.out, "lora_adapter.pt"))
                print(f"Checkpoint saved: {p}")

    avg = sum(losses) / max(len(losses), 1)
    print(f"\nTraining complete: {step} steps, avg loss={avg:.4f}")
    if args.smoke:
        print("SMOKE TEST PASSED")
    else:
        print(f"CP4.3 DONE — LoRA at {args.out}")


if __name__ == "__main__":
    main()
