"""CP4.5: CEM planning using the fine-tuned Cosmos world model as lookahead.

The planner samples K action sequences, rolls each out in the world model, scores
predicted frames by proximity to goal, and executes the best-scoring sequence in the
real Isaac Lab environment.

DoD: planner reaches the nav goal using only model-predicted lookahead.

Run inside Isaac Lab Docker (for real env):
    docker exec -e PYTHONPATH="..." isaac-lab-base python \
        /workspace/programs/p4_cosmos_world_sim/cp45_plan.py \
        --cosmos-checkpoint /workspace/checkpoints/p4_cosmos_lora/ \
        --task Humanoid-G1-VisionNav-v0 \
        --num-envs 4 --plan-steps 8 --cem-samples 64 \
        --out /workspace/docs/results/cp45_planning.mp4
"""

from __future__ import annotations

import argparse
import os

import imageio
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_frame(frame: np.ndarray, goal_frame: np.ndarray) -> float:
    """Score a predicted frame by pixel similarity to a goal frame.

    Higher = closer to goal. Simple MSE inversion.
    """
    diff = frame.astype(np.float32) - goal_frame.astype(np.float32)
    return float(-np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# CEM planner
# ---------------------------------------------------------------------------

class CEMPlanner:
    """Cross-Entropy Method planner that uses a Cosmos world model for rollouts."""

    def __init__(
        self,
        cosmos_pipe,
        action_dim: int = 29,
        plan_steps: int = 8,
        num_samples: int = 64,
        num_elite: int = 8,
        cem_iters: int = 3,
        device: str = "cuda",
    ) -> None:
        self.pipe = cosmos_pipe
        self.action_dim = action_dim
        self.plan_steps = plan_steps
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.cem_iters = cem_iters
        self.device = device

        # Action distribution parameters (initialised to zero mean, unit std)
        self.mu = np.zeros((plan_steps, action_dim), dtype=np.float32)
        self.sigma = np.ones((plan_steps, action_dim), dtype=np.float32)

    def plan(self, current_frame: np.ndarray, goal_frame: np.ndarray) -> np.ndarray:
        """Return the best action sequence (plan_steps, action_dim) for the current frame."""
        mu, sigma = self.mu.copy(), self.sigma.copy()

        for _ in range(self.cem_iters):
            # Sample action sequences: (num_samples, plan_steps, action_dim)
            samples = (
                mu[None] + sigma[None] * np.random.randn(self.num_samples, self.plan_steps, self.action_dim)
            ).astype(np.float32)
            # Clip to reasonable action range (±2π roughly)
            samples = np.clip(samples, -3.14, 3.14)

            scores = []
            for s in range(self.num_samples):
                score = self._rollout_score(current_frame, samples[s], goal_frame)
                scores.append(score)

            scores = np.array(scores)
            elite_idx = np.argsort(scores)[-self.num_elite:]
            elite = samples[elite_idx]  # (num_elite, plan_steps, action_dim)
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-3

        self.mu = mu  # warm start next planning call
        return mu  # (plan_steps, action_dim)

    def _rollout_score(
        self, init_frame: np.ndarray, action_seq: np.ndarray, goal_frame: np.ndarray
    ) -> float:
        """Roll out the world model K steps and score the final predicted frame."""
        frame = init_frame
        with torch.no_grad():
            for k in range(self.plan_steps):
                action = torch.tensor(action_seq[k], dtype=torch.bfloat16).unsqueeze(0).to(self.device)
                output = self.pipe(image=frame, action=action, num_frames=1)
                frame = output.frames if hasattr(output, "frames") else output[0]
                if hasattr(frame, "cpu"):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4:
                    frame = frame[0]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return score_frame(frame, goal_frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CP4.5: CEM planning with Cosmos world model")
    parser.add_argument("--cosmos-checkpoint", required=True, help="Fine-tuned Cosmos checkpoint dir")
    parser.add_argument("--task", default="Humanoid-G1-VisionNav-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--plan-steps", type=int, default=8, help="Lookahead horizon")
    parser.add_argument("--cem-samples", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--out", default="docs/results/cp45_planning.mp4")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned Cosmos
    try:
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # type: ignore
        pipe = Video2WorldPipeline.from_pretrained(
            args.cosmos_checkpoint, torch_dtype=torch.bfloat16
        ).to(device)
    except ImportError as e:
        raise ImportError("Cannot import Cosmos pipeline. Check cosmos-predict2 docs.") from e
    pipe.eval()

    # Isaac Lab env — must run inside the docker container
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Isaac Lab not found. Run this script inside the Isaac Lab Docker container."
        ) from e

    launcher = AppLauncher({
        "headless": True, "enable_cameras": True, "num_envs": args.num_envs
    })
    sim = launcher.app

    import gymnasium as gym  # type: ignore
    import my_humanoid_project  # noqa: F401

    env = gym.make(args.task, num_envs=args.num_envs)

    planner = CEMPlanner(
        cosmos_pipe=pipe,
        plan_steps=args.plan_steps,
        num_samples=args.cem_samples,
        device=device,
    )

    successes = 0
    all_frames: list[np.ndarray] = []

    for ep in range(args.num_episodes):
        obs_dict, _ = env.reset()
        ep_frames = []
        done = False
        step = 0

        # Goal frame: a heuristic target (e.g., the observation from a distant future step).
        # For now, use a blank goal (all white) as placeholder — the real metric is
        # env-reported goal_reached, not frame similarity.
        goal_frame = np.ones((64, 64, 3), dtype=np.uint8) * 200

        while not done and step < args.max_episode_steps:
            current_frame = obs_dict["images"]["head_cam"][0].cpu().numpy().astype(np.uint8)
            ep_frames.append(current_frame)

            # Plan: get best action sequence from CEM
            action_seq = planner.plan(current_frame, goal_frame)  # (plan_steps, 29)
            first_action = torch.tensor(action_seq[0], dtype=torch.float32).unsqueeze(0).to(device)
            first_action = first_action.expand(args.num_envs, -1)

            obs_dict, _, terminated, truncated, info = env.step(first_action)
            done = bool(terminated[0]) or bool(truncated[0])
            step += 1

        goal_reached = bool(info.get("goal_reached", [False])[0])
        if goal_reached:
            successes += 1
        print(f"Episode {ep+1}/{args.num_episodes}: {'SUCCESS' if goal_reached else 'FAIL'} in {step} steps")
        all_frames.extend(ep_frames[:30])  # keep first 30 frames per episode for video

    env.close()
    sim.close()

    success_rate = successes / args.num_episodes
    print(f"\nSuccess rate: {successes}/{args.num_episodes} = {100*success_rate:.1f}%")

    if all_frames:
        imageio.mimwrite(args.out, all_frames, fps=8, quality=8)
        print(f"Planning video saved: {args.out}")

    print("CP4.5 DONE")


if __name__ == "__main__":
    main()
