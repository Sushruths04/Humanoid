from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _latest_checkpoint(log_root: Path) -> Path:
    checkpoints = sorted(log_root.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found under: {log_root}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play back a trained Vision-VLA policy.")
    parser.add_argument(
        "--task",
        type=str,
        default=os.getenv("VISION_VLA_TASK", os.getenv("TASK", "Humanoid-G1-Vision-VLA-v0")),
    )
    parser.add_argument("--checkpoint", type=str, default=os.getenv("VISION_VLA_CHECKPOINT"))
    parser.add_argument(
        "--log-root",
        type=str,
        default=os.getenv("VISION_VLA_LOG_ROOT", os.path.join("logs", "rsl_rl", "g1_vla_vision_cnn")),
    )
    parser.add_argument("--num-envs", type=int, default=int(os.getenv("VISION_VLA_PLAY_ENVS", "1")))
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video-length", type=int, default=int(os.getenv("VISION_VLA_VIDEO_LENGTH", "400")))
    parser.add_argument("--disable-fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.video = args.video or os.getenv("VISION_VLA_VIDEO", "1") != "0"

    if args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    # Import task registration only after the simulator is up.
    import my_humanoid_project.tasks  # noqa: F401
    from my_humanoid_project.tasks.g1_vla_vision_cfg import G1VisionVLACnnRunnerCfg

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.log_root))
    video_folder = checkpoint.parent / "videos" / "play"
    video_folder.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    runner_cfg = G1VisionVLACnnRunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    steps = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        steps += 1
        if args.video and steps >= args.video_length:
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
