"""Evaluate a trained SEQUENTIAL-nav (SeqNav) policy.

Unlike evaluate.py (single commanded target), this rolls out the policy and, for
each env's first episode, records the first timestep at which each subgoal marker
is reached. It then reports full-sequence success + ordering accuracy via the
unit-tested programs.common.eval.metrics.sequence_eval_metrics.
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
    parser = argparse.ArgumentParser(description="Evaluate sequential-nav policy.")
    parser.add_argument("--task", type=str, default="Humanoid-G1-SeqNav-v0")
    parser.add_argument("--checkpoint", type=str, default=os.getenv("NAV_CHECKPOINT"))
    parser.add_argument("--log-root", type=str, default="logs/rsl_rl/g1_flat")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--reach-radius", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="docs/results/seqnav.md")
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
    from programs.common.eval.metrics import sequence_eval_metrics
    from programs.common.eval.report import write_results_markdown

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.log_root))
    print(f"[eval-seq] checkpoint: {checkpoint}")

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

    obs, _ = env.reset()
    # Snapshot the first-episode layout (env reshuffles these on reset).
    targets = base._seq_targets.clone()          # (n, num_subgoals) ordered marker ids
    markers = base._nav_markers_xy.clone()        # (n, num_markers, 2) local coords
    num_subgoals = int(targets.shape[1])
    idx = torch.arange(n, device=device)

    reach_steps = torch.full((n, num_subgoals), -1, dtype=torch.long, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    min_dist = torch.full((n, num_subgoals), float("inf"), device=device)
    start_xy = None
    cmd_norm_sum = torch.zeros(n, device=device)
    cmd_steps = 0

    def robot_xy():
        return base.scene["robot"].data.root_pos_w[:, :2] - base.scene.env_origins[:, :2]

    steps = 0
    while bool(active.any()) and steps < 2 * max_len:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        xy = robot_xy()
        if start_xy is None:
            start_xy = xy.clone()
        for k in range(num_subgoals):
            targ_k = markers[idx, targets[:, k]]              # (n, 2)
            dist_k = torch.linalg.norm(xy - targ_k, dim=-1)
            min_dist[active, k] = torch.minimum(min_dist[active, k], dist_k[active])
            newly = active & (reach_steps[:, k] < 0) & (dist_k < args.reach_radius)
            reach_steps[newly, k] = steps
        try:
            vc = base.command_manager.get_term("base_velocity").vel_command_b
            cmd_norm_sum += torch.linalg.norm(vc[:, :2], dim=-1)
            cmd_steps += 1
        except Exception:
            pass
        steps += 1
        # Freeze envs that just finished their first episode.
        active &= ~dones.bool()

    metrics = sequence_eval_metrics(reach_steps.cpu(), num_subgoals)
    metrics["reach_radius"] = args.reach_radius
    # Diagnostics: how close did the robot actually get, and did it move at all?
    md = min_dist.cpu()
    metrics["mean_min_dist_subgoal0"] = round(float(md[:, 0].mean()), 3)
    metrics["mean_min_dist_subgoal1"] = round(float(md[:, 1].mean()), 3)
    metrics["reach0_rate@1.0m"] = round(float((md[:, 0] < 1.0).float().mean()), 3)
    metrics["reach0_rate@2.0m"] = round(float((md[:, 0] < 2.0).float().mean()), 3)
    end_xy = robot_xy().cpu()
    disp = torch.linalg.norm(end_xy - start_xy.cpu(), dim=-1)
    metrics["mean_robot_displacement"] = round(float(disp.mean()), 3)
    metrics["mean_cmd_vel_norm"] = round(float((cmd_norm_sum / max(cmd_steps, 1)).mean()), 3)
    metrics["checkpoint"] = str(checkpoint)
    print("[eval-seq] metrics:", metrics)

    target = Path("/workspace") / args.out
    try:
        write_results_markdown(metrics, str(target), title=f"Eval: {args.task}")
        print(f"[eval-seq] wrote {target}")
    except Exception as exc:  # noqa: BLE001
        write_results_markdown(metrics, args.out, title=f"Eval: {args.task}")
        print(f"[eval-seq] wrote {args.out} ({exc})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
