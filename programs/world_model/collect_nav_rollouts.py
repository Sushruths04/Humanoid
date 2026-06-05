"""Collect (obs, action, reward) rollouts from a trained nav policy for P2.

Runs the policy in Humanoid-G1-CommandNav-v0 (or any nav task) for N episodes,
records per-step tensors, and saves a compact .pt dataset.

Usage:
    python -m programs.world_model.collect_nav_rollouts \
        --task Humanoid-G1-CommandNav-v0 \
        --checkpoint programs/checkpoints/g1_commandnav/model_499.pt \
        --num-envs 64 --num-episodes 200 \
        --out programs/data/nav_rollouts_commandnav.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect nav rollouts for world-model training.")
    parser.add_argument("--task", type=str, default="Humanoid-G1-CommandNav-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-episodes", type=int, default=200,
                        help="Target number of complete episodes to collect")
    parser.add_argument("--max-ep-steps", type=int, default=None,
                        help="Max steps per episode (defaults to env max_episode_length)")
    parser.add_argument("--obs-keys", type=str, default="nav",
                        choices=["nav", "full"],
                        help="nav = 4-dim nav obs only; full = full policy obs")
    parser.add_argument("--out", type=str, default="programs/data/nav_rollouts.pt")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import importlib.metadata as _metadata
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import my_humanoid_project.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    base = env.unwrapped
    n = base.num_envs
    max_ep = args.max_ep_steps or int(base.max_episode_length)

    collected = {"obs": [], "action": [], "reward": []}
    ep_obs, ep_act, ep_rew = [], [], []
    episodes_done = 0

    obs, _ = env.reset()

    def _to_tensor(o):
        """Extract flat (N, obs_dim) tensor from obs — handles TensorDict and plain tensor."""
        if isinstance(o, torch.Tensor):
            return o
        # RSL-RL TensorDict: key is "obs" or first key
        if hasattr(o, "get"):
            t = o.get("obs", None)
            if t is None:
                t = next(iter(o.values()))
            return t
        return torch.as_tensor(o)

    obs_t = _to_tensor(obs)
    print(f"[collect] obs shape: {obs_t.shape}")

    while episodes_done < args.num_episodes:
        with torch.inference_mode():
            action = policy(obs)
        next_obs, reward, dones, _ = env.step(action)

        obs_t = _to_tensor(obs)
        if args.obs_keys == "nav":
            # Last 4 dims of policy obs are nav_command_obs: one-hot + rel_xy
            nav_obs = obs_t[:, -4:]
        else:
            nav_obs = obs_t

        ep_obs.append(nav_obs.cpu())
        ep_act.append(action.cpu())
        ep_rew.append(reward.cpu())

        done_envs = dones.bool().nonzero(as_tuple=False).flatten().tolist()
        for env_i in done_envs:
            # Slice out this env's trajectory from current episode buffers
            t = len(ep_obs)
            obs_ep = torch.stack([o[env_i] for o in ep_obs])       # (T, obs_dim)
            act_ep = torch.stack([a[env_i] for a in ep_act])       # (T, act_dim)
            rew_ep = torch.stack([r[env_i] for r in ep_rew])       # (T,)
            collected["obs"].append(obs_ep)
            collected["action"].append(act_ep)
            collected["reward"].append(rew_ep)
            episodes_done += 1
            if episodes_done % 20 == 0:
                print(f"[collect] {episodes_done}/{args.num_episodes} episodes")

        obs = next_obs

        if dones.bool().all() and episodes_done < args.num_episodes:
            ep_obs, ep_act, ep_rew = [], [], []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "obs": collected["obs"],
        "action": collected["action"],
        "reward": collected["reward"],
        "task": args.task,
        "checkpoint": args.checkpoint,
        "num_episodes": episodes_done,
    }, out)
    print(f"[collect] saved {episodes_done} episodes → {out}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
