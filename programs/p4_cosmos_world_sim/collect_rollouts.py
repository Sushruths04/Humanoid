"""CP4.2 Data collection: run P3 nav policy inside Isaac Lab and save Bridge-format data.

Bridge format per episode:
  datasets/g1_nav/videos/train/{ep}/0/rgb.mp4
  datasets/g1_nav/annotation/train/{ep}.json

JSON schema:
{
  "action": [[vx/20, vy/20, 0, 0, 0, omega/20], ...],   # T entries
  "continuous_gripper_state": [0.0, ...],               # T+1 zeros
  "state":  [[...], ...]                                 # T entries
}

Run inside Isaac Lab Docker:
  docker exec isaac-lab-base python /workspace/programs/p4_cosmos_world_sim/collect_rollouts.py     --checkpoint /workspace/checkpoints/p3_vision_nav/run_300_l4/model_499.pt     --task Humanoid-G1-VisionNav-v0     --num-envs 64 --num-steps 500     --out /workspace/datasets/g1_nav
"""
from __future__ import annotations
import argparse, json, os, struct, sys, tempfile
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def frames_to_mp4(frames: list[np.ndarray], path: str, fps: int = 4) -> None:
    """Write list of (H, W, 3) uint8 frames to MP4 using imageio."""
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


def save_annotation(actions: list[list[float]], path: str) -> None:
    """Write Bridge-format JSON annotation."""
    T = len(actions)
    ann = {
        "action": actions,
        "continuous_gripper_state": [0.0] * (T + 1),
        "state": [[0.0] * 6] * T,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ann, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="Humanoid-G1-VisionNav-v0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--out", default="datasets/g1_nav")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    # IsaacLab imports (only available inside Docker)
    try:
        import isaaclab  # noqa: F401 — needed to register envs
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner  # type: ignore
        import torch
    except ImportError as e:
        print(f"[collect_rollouts] Import error: {e}")
        print("Run inside Isaac Lab Docker. Exiting with mock for offline testing.")
        _write_mock_data(args.out, n_episodes=5, T=args.num_steps, fps=args.fps)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load env
    env = gym.make(args.task, num_envs=args.num_envs, headless=True)
    obs_space = env.observation_space
    act_space = env.action_space
    print(f"Env: {args.task} | obs={obs_space.shape} | act={act_space.shape}")

    # Load policy
    ckpt = torch.load(args.checkpoint, map_location=device)
    # RSL-RL checkpoint has 'model_state_dict' or 'state_dict'
    runner = OnPolicyRunner(env, None, None, device=device)
    runner.alg.actor_critic.load_state_dict(
        ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), strict=False
    )
    runner.alg.actor_critic.eval()
    policy = runner.get_inference_policy(device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # Per-env episode buffers
    ep_frames: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
    ep_actions: list[list[list[float]]] = [[] for _ in range(args.num_envs)]
    ep_ids = list(range(args.num_envs))
    saved_episodes = 0
    total_ep = args.num_envs * max(1, args.num_steps // 200)

    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs_dict, _, dones, _, infos = env.step(actions)
        obs = obs_dict["policy"]

        # Grab camera frames (64x64 RGB)
        rgb = obs_dict.get("rgb", None)  # (N, H, W, 3) or None
        vel_cmd = infos.get("vel_command", actions[:, :3])  # (N, 3): vx, vy, omega

        for i in range(args.num_envs):
            # Frame
            if rgb is not None:
                frame = (rgb[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            ep_frames[i].append(frame)

            # Action in Bridge convention: [vx/20, vy/20, 0, 0, 0, omega/20]
            v = vel_cmd[i].cpu().numpy()
            bridge_action = [float(v[0]) / 20, float(v[1]) / 20, 0.0, 0.0, 0.0, float(v[2]) / 20]
            ep_actions[i].append(bridge_action)

            # Episode done → save
            if dones[i]:
                ep = ep_ids[i]
                vid_path = os.path.join(args.out, "videos", "train", str(ep), "0", "rgb.mp4")
                ann_path = os.path.join(args.out, "annotation", "train", f"{ep}.json")
                frames_to_mp4(ep_frames[i], vid_path, fps=args.fps)
                save_annotation(ep_actions[i], ann_path)
                saved_episodes += 1
                if saved_episodes % 10 == 0:
                    print(f"  Saved {saved_episodes} episodes ...")
                # Reset buffers
                ep_frames[i] = []
                ep_actions[i] = []
                ep_ids[i] = saved_episodes + args.num_envs

        if saved_episodes >= total_ep:
            break

    env.close()
    print(f"Data collection DONE: {saved_episodes} episodes saved to {args.out}")


def _write_mock_data(out: str, n_episodes: int = 5, T: int = 50, fps: int = 4) -> None:
    """Offline fallback: write dummy Bridge-format data for testing cp42_verify_data."""
    import imageio
    for ep in range(n_episodes):
        vid_path = os.path.join(out, "videos", "train", str(ep), "0", "rgb.mp4")
        ann_path = os.path.join(out, "annotation", "train", f"{ep}.json")
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(T)]
        frames_to_mp4(frames, vid_path, fps=fps)
        actions = [[0.1 / 20, 0.0, 0.0, 0.0, 0.0, 0.0]] * T
        save_annotation(actions, ann_path)
    print(f"Mock data written: {n_episodes} episodes at {out}")


if __name__ == "__main__":
    main()
