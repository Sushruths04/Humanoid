"""CP4.5: CEM planning demo using the fine-tuned Cosmos world model.

Standalone (no Isaac Lab required). Uses Bridge data as context.
CEM finds action sequences that minimize MSE between imagined future
and a goal state. Demonstrates the world model supports planning.

DoD: CEM MSE < random baseline; 3-column demo video saved
     (conditioning | random imagined | CEM planned).
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


def prepare_video_tensor(frames):
    """Returns vid [1,3,T_pad,H,W] bfloat16, T_pad."""
    import torch
    import numpy as np
    T = min(len(frames), 12)
    frames = frames[:T]
    pad = (4 - (T - 1) % 4) % 4
    if pad > 0:
        frames = np.concatenate([frames, np.tile(frames[-1:], (pad, 1, 1, 1))], axis=0)
    vid = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
    vid = (vid / 255.0) * 2.0 - 1.0
    return vid.to(torch.bfloat16), frames.shape[0]


def score_actions(pipe, cond_latent, mask, H_vid, W_vid, act_np):
    """
    Single-step x0 prediction at t=0.5 (sigma=1.0) — fast discriminative score.
    Returns latent MSE between predicted x0 and cond_latent (lower = more similar to start).
    act_np: [12, 7] numpy float32
    """
    import torch
    from cosmos_predict2.conditioner import ActionCondition, DataType

    B = 1
    act_t = torch.from_numpy(act_np[None]).to(cond_latent.device, torch.bfloat16)
    condition = ActionCondition(
        crossattn_emb=torch.zeros(B, 256, 1024, device=cond_latent.device, dtype=pipe.precision),
        data_type=DataType.VIDEO,
        padding_mask=torch.zeros(B, 1, H_vid, W_vid, device=cond_latent.device, dtype=pipe.precision),
        fps=None,
        use_video_condition=True,
        gt_frames=cond_latent,
        condition_video_input_mask_B_C_T_H_W=mask,
        action=act_t,
    )
    sigma_t = torch.full((B,), 1.0, device=cond_latent.device, dtype=pipe.precision)
    xt = cond_latent + torch.randn_like(cond_latent)  # t=0.5: xt ≈ 0.5*x0 + 0.5*noise
    with torch.no_grad():
        pred = pipe.denoise(xt, sigma_t, condition)
    return pred.x0  # [1, C, T_lat, H_lat, W_lat]


def generate_frames_ode(pipe, cond_latent, mask, H_vid, W_vid, act_np, n_steps: int = 10):
    """Full Euler ODE generation for final visualization."""
    import torch
    import numpy as np
    from cosmos_predict2.conditioner import ActionCondition, DataType

    B = 1
    act_t = torch.from_numpy(act_np[None]).to(cond_latent.device, torch.bfloat16)
    condition = ActionCondition(
        crossattn_emb=torch.zeros(B, 256, 1024, device=cond_latent.device, dtype=pipe.precision),
        data_type=DataType.VIDEO,
        padding_mask=torch.zeros(B, 1, H_vid, W_vid, device=cond_latent.device, dtype=pipe.precision),
        fps=None,
        use_video_condition=True,
        gt_frames=cond_latent,
        condition_video_input_mask_B_C_T_H_W=mask,
        action=act_t,
    )
    x_t = torch.randn_like(cond_latent)
    ts = torch.linspace(0.99, 0.01, n_steps + 1, device=cond_latent.device, dtype=pipe.precision)

    with torch.no_grad():
        for i in range(n_steps):
            t_c, t_n = float(ts[i]), float(ts[i + 1])
            sigma_t = torch.full((B,), t_c / (1.0 - t_c + 1e-8), device=x_t.device, dtype=pipe.precision)
            pred = pipe.denoise(x_t, sigma_t, condition)
            x_t = x_t + (t_n - t_c) * (x_t - pred.x0) / (t_c + 1e-8)

        decoded = pipe.tokenizer.decode(x_t)  # [1, 3, T, H, W] in [-1, 1]

    frames = decoded[0].permute(1, 2, 3, 0).cpu().float().numpy()
    return np.clip((frames + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)


def run_cem(pipe, cond_latent, mask, H_vid, W_vid, goal_latent,
            num_samples, num_elite, cem_iters):
    """Cross-Entropy Method: find actions minimising MSE to goal_latent."""
    import torch
    import numpy as np

    mu = np.zeros((12, 7), dtype=np.float32)
    sigma = np.ones((12, 7), dtype=np.float32) * 2.0

    for it in range(cem_iters):
        samples = mu[None] + sigma[None] * np.random.randn(num_samples, 12, 7).astype(np.float32)
        samples = np.clip(samples, -6.0, 6.0)

        scores = []
        for s in samples:
            pred_x0 = score_actions(pipe, cond_latent, mask, H_vid, W_vid, s)
            mse = float(torch.mean((pred_x0 - goal_latent) ** 2).cpu())
            scores.append(-mse)  # negative MSE → higher is better

        elite_idx = np.argsort(scores)[-num_elite:]
        elite = samples[elite_idx]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-3
        print(f"    CEM iter {it+1}/{cem_iters}: best score={scores[elite_idx[-1]]:.6f}")

    return mu  # [12, 7] best action sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--checkpoint", default=LORA_CKPT)
    parser.add_argument("--model-dir", default=os.path.dirname(BASE_CKPT))
    parser.add_argument("--cem-samples", type=int, default=16)
    parser.add_argument("--cem-elite", type=int, default=4)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--out", default=f"{REPO}/docs/results/cp45_planning.mp4")
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

    misc.set_random_seed(seed=42, by_rank=True)
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
    pipe.dit.load_state_dict(lora_state, strict=False)
    pipe.dit.eval()

    ann_dir = Path(args.data) / "annotation" / "train"
    ep_ids = sorted([int(p.stem) for p in ann_dir.glob("*.json")])

    planned_mses, random_mses = [], []
    video_frames = []

    for trial in range(min(args.num_trials, len(ep_ids) - 1)):
        cond_ep = ep_ids[trial]
        goal_ep = ep_ids[trial + 1]
        print(f"\nTrial {trial+1}/{args.num_trials}: cond={cond_ep} goal={goal_ep}")

        cond_frames, cond_actions = load_bridge_episode(args.data, cond_ep)
        goal_frames, _ = load_bridge_episode(args.data, goal_ep)

        cond_vid, T_pad = prepare_video_tensor(cond_frames)
        goal_vid, _ = prepare_video_tensor(goal_frames)

        with torch.no_grad():
            cond_latent = pipe.tokenizer.encode(cond_vid)
            goal_latent = pipe.tokenizer.encode(goal_vid)

        B, C, T_lat, H_lat, W_lat = cond_latent.shape
        H_vid, W_vid = cond_vid.shape[-2], cond_vid.shape[-1]
        mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=cond_latent.device, dtype=cond_latent.dtype)
        mask[:, :, :1] = 1.0

        # CEM planning
        planned_actions = run_cem(
            pipe, cond_latent, mask, H_vid, W_vid, goal_latent,
            args.cem_samples, args.cem_elite, args.cem_iters,
        )

        # Random baseline
        random_actions = np.random.randn(12, 7).astype(np.float32) * 0.5

        # Evaluate both with full ODE
        print(f"  Generating planned trajectory ({args.denoise_steps} steps) ...")
        planned_gen = generate_frames_ode(pipe, cond_latent, mask, H_vid, W_vid,
                                          planned_actions, args.denoise_steps)
        print(f"  Generating random trajectory ...")
        random_gen = generate_frames_ode(pipe, cond_latent, mask, H_vid, W_vid,
                                         random_actions, args.denoise_steps)

        # Score vs goal (decode goal for comparison)
        goal_dec = pipe.tokenizer.decode(goal_latent)
        goal_frames_dec = np.clip(
            (goal_dec[0].permute(1, 2, 3, 0).cpu().float().numpy() + 1.0) / 2.0 * 255, 0, 255
        ).astype(np.uint8)
        cond_dec = pipe.tokenizer.decode(cond_latent)
        cond_frames_dec = np.clip(
            (cond_dec[0].permute(1, 2, 3, 0).cpu().float().numpy() + 1.0) / 2.0 * 255, 0, 255
        ).astype(np.uint8)

        T_cmp = min(planned_gen.shape[0], random_gen.shape[0], goal_frames_dec.shape[0]) - 1

        def mse_to_goal(gen, goal):
            g = goal[1:T_cmp+1].astype(np.float32)
            p = gen[1:T_cmp+1].astype(np.float32)
            if g.shape != p.shape:
                import cv2
                p = np.stack([cv2.resize(f, (g.shape[2], g.shape[1])) for f in p])
            return float(np.mean((p - g) ** 2))

        p_mse = mse_to_goal(planned_gen, goal_frames_dec)
        r_mse = mse_to_goal(random_gen, goal_frames_dec)
        planned_mses.append(p_mse)
        random_mses.append(r_mse)
        improvement = (r_mse - p_mse) / (r_mse + 1e-8) * 100
        print(f"  MSE to goal: planned={p_mse:.2f} | random={r_mse:.2f} | improvement={improvement:.1f}%")

        # Build comparison video: 3 cols (cond | random | planned), 4 frames each
        H_g, W_g = goal_frames_dec.shape[1], goal_frames_dec.shape[2]
        N = min(4, cond_frames_dec.shape[0], random_gen.shape[0], planned_gen.shape[0])
        for fi in range(N):
            def resize_to(frame, H, W):
                if frame.shape[0] == H and frame.shape[1] == W:
                    return frame
                import cv2
                return cv2.resize(frame.astype(np.uint8), (W, H))

            row = np.concatenate([
                resize_to(cond_frames_dec[fi], H_g, W_g),
                resize_to(random_gen[fi], H_g, W_g),
                resize_to(planned_gen[fi], H_g, W_g),
            ], axis=1)
            video_frames.append(row)

    print(f"\n===== CP4.5 CEM Planning Results =====")
    print(f"Trials: {len(planned_mses)}")
    print(f"Avg planned MSE: {np.mean(planned_mses):.2f}")
    print(f"Avg random  MSE: {np.mean(random_mses):.2f}")
    improvement = (np.mean(random_mses) - np.mean(planned_mses)) / (np.mean(random_mses) + 1e-8) * 100
    print(f"CEM improvement over random: {improvement:.1f}%")
    print("Columns in video: [conditioning | random] | [CEM planned]")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if video_frames:
        imageio.mimwrite(args.out, video_frames, fps=2, quality=8)
        print(f"Planning demo video saved: {args.out}")

    print("CP4.5 DONE")


if __name__ == "__main__":
    main()
