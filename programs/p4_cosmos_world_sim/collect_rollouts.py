"""CP4.2 data collection: run P3 nav policy in Isaac Lab, save (frame_t, action_t, frame_t+1) triplets.

Run inside Isaac Lab Docker:
    docker exec -e PYTHONPATH="..." isaac-lab-base python \
        /workspace/programs/p4_cosmos_world_sim/collect_rollouts.py \
        --checkpoint /workspace/checkpoints/p3_vision_nav/run_300_l4/model_499.pt \
        --task Humanoid-G1-VisionNav-v0 \
        --num-envs 64 --num-steps 500 \
        --out /workspace/datasets/g1_nav_cosmos.h5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch


def _load_rsl_policy(checkpoint_path: str, device: str):
    """Load RSL-RL actor from a saved checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # RSL-RL checkpoints store model weights under 'model_state_dict'
    # The actor network is extracted at inference time via the runner.
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect P3 nav rollouts for Cosmos training")
    parser.add_argument("--checkpoint", required=True, help="Path to model_499.pt")
    parser.add_argument("--task", default="Humanoid-G1-VisionNav-v0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--out", required=True, help="Output HDF5 path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Isaac Lab imports (only available inside the docker container)
    try:
        import isaaclab.app  # noqa: F401 — needed to boot Isaac Sim
    except ImportError as e:
        raise ImportError(
            "Isaac Lab not found. Run this script inside the Isaac Lab Docker container."
        ) from e

    from isaaclab.app import AppLauncher  # type: ignore

    launcher = AppLauncher({"headless": True, "enable_cameras": True, "num_envs": args.num_envs})
    sim = launcher.app

    # Late import after sim starts
    import gymnasium as gym  # type: ignore
    from rsl_rl.runners import OnPolicyRunner  # type: ignore

    import my_humanoid_project  # noqa: F401 — registers tasks

    env = gym.make(args.task, num_envs=args.num_envs)

    # Load the policy via RSL-RL runner (mirrors how custom_play.py does it)
    runner = OnPolicyRunner(env, {"policy": {"class_name": "ActorCritic"}}, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    policy.eval()

    frames_t: list[np.ndarray] = []
    actions_t: list[np.ndarray] = []
    frames_t1: list[np.ndarray] = []

    obs_dict, _ = env.reset()

    for step in range(args.num_steps):
        # obs_dict["images"]["head_cam"] shape: (num_envs, 64, 64, 3) uint8
        img_t = obs_dict["images"]["head_cam"].cpu().numpy().astype(np.uint8)

        obs_flat = torch.cat(
            [obs_dict["policy"], obs_dict.get("images_flat", torch.zeros(args.num_envs, 0, device=device))],
            dim=-1,
        )
        with torch.no_grad():
            action = policy(obs_flat)

        obs_dict, _, _, _, _ = env.step(action)

        img_t1 = obs_dict["images"]["head_cam"].cpu().numpy().astype(np.uint8)
        act_np = action.cpu().numpy().astype(np.float32)

        frames_t.append(img_t)     # (num_envs, 64, 64, 3)
        actions_t.append(act_np)   # (num_envs, 29)
        frames_t1.append(img_t1)   # (num_envs, 64, 64, 3)

        if step % 50 == 0:
            total = (step + 1) * args.num_envs
            print(f"Step {step+1}/{args.num_steps}  triplets so far: {total}")

    env.close()
    sim.close()

    # Stack: (num_steps * num_envs, ...)
    ft = np.concatenate(frames_t, axis=0)
    at = np.concatenate(actions_t, axis=0)
    ft1 = np.concatenate(frames_t1, axis=0)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("frame_t", data=ft, compression="gzip", chunks=(256, 64, 64, 3))
        f.create_dataset("action_t", data=at, compression="gzip", chunks=(256, 29))
        f.create_dataset("frame_t1", data=ft1, compression="gzip", chunks=(256, 64, 64, 3))

    n = ft.shape[0]
    print(f"Saved {n} triplets to {args.out}")
    print(f"  frame_t:  {ft.shape}  {ft.dtype}")
    print(f"  action_t: {at.shape}  {at.dtype}")
    print(f"  frame_t1: {ft1.shape}  {ft1.dtype}")


if __name__ == "__main__":
    main()
