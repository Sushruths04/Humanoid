"""Evaluate a trained command-nav policy: measures commanded-target success.

Rolls out the policy and records, per episode (first episode per env):
  - reached: robot came within reach_radius of the COMMANDED marker
  - fell: episode terminated before max length (early termination)
  - final_distance / episode_length / command id
Then aggregates with the unit-tested metrics and writes a markdown report.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _latest_checkpoint(log_root: Path) -> Path:
    cks = sorted(log_root.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise FileNotFoundError(f"No checkpoint under {log_root}")
    return cks[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate command-nav policy.")
    parser.add_argument("--task", type=str, default="Humanoid-G1-CommandNav-v0")
    parser.add_argument("--checkpoint", type=str, default=os.getenv("NAV_CHECKPOINT"))
    parser.add_argument("--log-root", type=str, default="logs/rsl_rl/g1_flat")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--reach-radius", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="docs/results/p0_baseline.md")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import importlib.metadata as _metadata
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import my_humanoid_project.tasks  # noqa: F401  (registers tasks)
    from programs.common.eval.metrics import compute_episode_metrics, success_rate_by_command
    from programs.common.eval.report import write_results_markdown

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.log_root))
    print(f"[eval] checkpoint: {checkpoint}")

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=args.device)

    base = env.unwrapped
    n, device = base.num_envs, base.device
    max_len = int(base.max_episode_length)

    active = torch.ones(n, dtype=torch.bool, device=device)
    reached = torch.zeros(n, dtype=torch.bool, device=device)
    ep_len = torch.zeros(n, dtype=torch.long, device=device)

    def commanded_dist():
        idx = torch.arange(n, device=device)
        target = base._nav_markers_xy[idx, base._nav_target_ids]
        xy = base.scene["robot"].data.root_pos_w[:, :2] - base.scene.env_origins[:, :2]
        return torch.linalg.norm(xy - target, dim=-1)

    obs, _ = env.reset()
    episode_command = base._nav_target_ids.clone()
    prev_d = commanded_dist()

    out = {"reached": [], "fell": [], "final_distance": [], "episode_length": [], "command": []}

    steps = 0
    while bool(active.any()) and steps < 2 * max_len:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        steps += 1
        d = commanded_dist()
        ep_len += active.long()
        reached |= active & (prev_d < args.reach_radius)

        newly = (dones.bool() & active).nonzero(as_tuple=False).flatten()
        for i in newly.tolist():
            out["reached"].append(bool(reached[i]))
            out["fell"].append(bool(ep_len[i] < max_len - 1))
            out["final_distance"].append(float(prev_d[i]))
            out["episode_length"].append(int(ep_len[i]))
            out["command"].append(int(episode_command[i]))
            active[i] = False

        prev_d = d

    # Any env that never terminated counts as a survived (timeout) episode.
    for i in active.nonzero(as_tuple=False).flatten().tolist():
        out["reached"].append(bool(reached[i]))
        out["fell"].append(False)
        out["final_distance"].append(float(prev_d[i]))
        out["episode_length"].append(int(ep_len[i]))
        out["command"].append(int(episode_command[i]))

    reached_t = torch.tensor(out["reached"])
    metrics = compute_episode_metrics(
        reached_t, torch.tensor(out["fell"]),
        torch.tensor(out["final_distance"]), torch.tensor(out["episode_length"]),
    )
    num_markers = int(base._nav_markers_xy.shape[1])
    by_cmd = success_rate_by_command(reached_t, torch.tensor(out["command"]), num_markers)
    metrics["success_by_command"] = [round(float(x), 3) for x in by_cmd]
    metrics["checkpoint"] = str(checkpoint)

    print("[eval] metrics:", metrics)
    repo_out = Path("/workspace/programs").parent / args.out if not os.path.isabs(args.out) else Path(args.out)
    # Write under the repo (mounted at /workspace); fall back to cwd.
    target = Path("/workspace") / args.out
    try:
        write_results_markdown(metrics, str(target), title="P0 Command-Nav Baseline")
        print(f"[eval] wrote {target}")
    except Exception as exc:  # noqa: BLE001
        write_results_markdown(metrics, args.out, title="P0 Command-Nav Baseline")
        print(f"[eval] wrote {args.out} ({exc})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
