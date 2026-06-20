"""Record a rollout video from a checkpoint.

Usage (inside container):
    python play.py --checkpoint runs/wakeboard_stage1/model_300.pt --v_pull_kmh 10 --episodes 3 --out results/rollout.mp4

Needs --enable_cameras for viewport capture. Falls back to tensorboard-only if
viewport capture is unavailable.
"""
from __future__ import annotations
import argparse, os, sys, subprocess, tempfile, glob
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--v_pull_kmh", type=float, default=10.0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--out", default="results/rollout.mp4")
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg, T_SUCCESS
    from src.rope_model import kmh_to_ms

    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 8.0
    env = WakeboardStartEnv(env_cfg)
    env.rope.set_v_pull(kmh_to_ms(args.v_pull_kmh))

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    rl_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(rl_env, _min_cfg(), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(env.device))

    # Try offscreen rendering via omni.replicator render products
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Sdf, Gf

        # Create a camera for rendering
        camera = rep.create.camera(position=(0, -3, 1.5), look_at=(0, 0, 0.7))
        render_product = rep.create.render_product(camera, resolution=(640, 480))

        frame_dir = os.path.join(os.path.dirname(args.out), "frames")
        os.makedirs(frame_dir, exist_ok=True)

        obs, _ = env.reset()
        collected = 0
        ep_steps = 0
        frames_captured = []

        for step in range(400):
            with torch.no_grad():
                act = policy(obs)
            result = env.step(act)
            if len(result) >= 3:
                obs = result[0]
            ep_steps += 1

            # Capture frame every 4th step (30fps -> 7.5fps effective)
            if step % 4 == 0:
                annotator = rep.AnnotatorRegistry.get.annotators["rgb"]
                annotator.attach([render_product])
                rep.step.render()
                frame_data = annotator.get_data()
                if frame_data is not None:
                    import numpy as np
                    from PIL import Image
                    fname = os.path.join(frame_dir, f"frame_{step:05d}.png")
                    img = Image.fromarray(np.array(frame_data))
                    img.save(fname)
                    frames_captured.append(fname)

            done_idx = result[2].nonzero().flatten() if len(result) > 2 and isinstance(result[2], torch.Tensor) else []
            for i in (done_idx.tolist() if hasattr(done_idx, 'tolist') else []):
                collected += 1
                ep_steps = 0
                if collected >= args.episodes:
                    break
            if collected >= args.episodes:
                break

        if frames_captured:
            _encode_frames(frame_dir, args.out)
            print(f"[play] Video saved to {args.out} ({len(frames_captured)} frames)")
        else:
            print("[play] No frames captured, falling back to trace mode")
            _run_without_video(env, policy, args)

    except Exception as e:
        print(f"[play] Offscreen rendering failed: {e}")
        print("[play] Falling back to trace mode")
        _run_without_video(env, policy, args)

    env.close()
    simulation_app.close()

def _run_without_video(env, policy, args):
    """Run rollout and save observation sequence as numpy (no video)."""
    import torch, json
    obs, _ = env.reset()
    frames = {"pelvis_z": [], "board_pitch": [], "fell": [], "reward": []}
    collected = 0
    for step in range(800):
        with torch.no_grad():
            act = policy(obs)
        result = env.step(act)
        if len(result) == 5:
            obs, rewards, dones, truncated, infos = result
        elif len(result) == 4:
            obs, rewards, dones, infos = result
        else:
            obs = result[0]
        h = env.scene["robot"].data.root_pos_w[:, 2].mean().item()
        bp = env._board_pitch.mean().item() * 180 / 3.14159
        frames["pelvis_z"].append(h)
        frames["board_pitch"].append(bp)
        frames["fell"].append(env._fall_event.float().mean().item())
        frames["reward"].append(rewards.mean().item())
        done_idx = dones.nonzero().flatten()
        for i in done_idx.tolist():
            collected += 1
            if collected >= args.episodes:
                break
        if collected >= args.episodes:
            break
    # Save as JSON for analysis
    out_json = args.out.replace(".mp4", "_trace.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(frames, f)
    print(f"[play] Saved trace to {out_json} ({len(frames['pelvis_z'])} steps)")
    print(f"[play] Pelvis Z: {frames['pelvis_z'][-10:]}")
    print(f"[play] Board pitch: {frames['board_pitch'][-10:]}")
    print(f"[play] Reward: {frames['reward'][-10:]}")

def _encode_frames(frame_dir, out_path):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frames:
        print("[play] No frames captured, skipping video encode")
        return
    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(frame_dir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)
    print(f"[play] Video saved to {out_path}")

def _min_cfg():
    """Minimal RSL-RL runner config for inference (matches train.py build_rsl_rl_cfg)."""
    hidden = [256, 256]
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    runner = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1,
        save_interval=1,
        experiment_name="play",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=hidden, activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=hidden, activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=5e-4, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )
    rl_cfg = runner.to_dict()
    for grp in ("actor", "critic"):
        for dep in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            if isinstance(rl_cfg.get(grp), dict):
                rl_cfg[grp].pop(dep, None)
    return rl_cfg

if __name__ == "__main__":
    main()
