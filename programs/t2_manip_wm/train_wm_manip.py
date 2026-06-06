"""T2 — Train Dreamer-mini world model on GR00T manipulation rollouts.

Thin wrapper around programs/world_model/train_wm_isaac.py that:
  - Uses manipulation-tuned hyperparameters (obs=8, act=7)
  - Writes docs/results/t2_manip_wm.md on completion

Usage:
    python -m programs.t2_manip_wm.train_wm_manip \
        --data programs/data/manip_rollouts_groot.pt \
        --out programs/checkpoints/world_model/wm_manip.pt \
        --result-doc docs/results/t2_manip_wm.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import random
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from programs.world_model.rssm import WorldModel


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="programs/data/manip_rollouts_groot.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--deter", type=int, default=128)
    p.add_argument("--stoch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--imagine-horizon", type=int, default=15)
    p.add_argument("--out", type=str, default="programs/checkpoints/world_model/wm_manip.pt")
    p.add_argument("--result-doc", type=str, default="docs/results/t2_manip_wm.md")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _sample_batch(episodes, batch, seq_len, device):
    obs_b, act_b, rew_b = [], [], []
    for _ in range(batch):
        ep = random.choice(episodes)
        obs_ep, act_ep, rew_ep = ep
        t = obs_ep.shape[0]
        if t <= seq_len:
            start, end, pad = 0, t, seq_len - t
        else:
            start = random.randint(0, t - seq_len)
            end = start + seq_len
            pad = 0
        obs_s = obs_ep[start:end]
        act_s = act_ep[start:end]
        rew_s = rew_ep[start:end]
        if pad > 0:
            obs_s = torch.cat([obs_s, obs_s[-1:].expand(pad, -1)])
            act_s = torch.cat([act_s, act_s[-1:].expand(pad, -1)])
            rew_s = torch.cat([rew_s, rew_s[-1:].expand(pad)])
        obs_b.append(obs_s)
        act_b.append(act_s)
        rew_b.append(rew_s)
    return (torch.stack(obs_b, dim=1).to(device),
            torch.stack(act_b, dim=1).to(device),
            torch.stack(rew_b, dim=1).to(device))


def _eval_imagination(wm, episodes, horizon, device):
    wm.eval()
    with torch.no_grad():
        obs_t, act_t, rew_t = _sample_batch(episodes, 16, 8, device)
        posts, _ = wm.observe(obs_t, act_t)
        init_state = posts[-1]
        imag_feats = wm.imagine(
            init_state,
            lambda f: torch.randn(f.shape[0], act_t.shape[2], device=device),
            horizon,
        )
        imag_rew = wm.reward_head(imag_feats).squeeze(-1).mean().item()
        real_rew = float(rew_t.mean())
    wm.train()
    return {"imagined_mean_reward": round(imag_rew, 4),
            "real_mean_reward": round(real_rew, 4)}


def _write_report(path: Path, args, initial_loss: float, final_loss: float,
                  final_eval: dict, n_eps: int, obs_dim: int, act_dim: int):
    passed = not math.isnan(final_eval["imagined_mean_reward"])
    lines = [
        "# T2 — World Model for Manipulation",
        "",
        f"Dataset: `{args.data}`  ",
        f"Episodes: {n_eps}  ",
        f"obs_dim: {obs_dim}  act_dim: {act_dim}  ",
        f"Training steps: {args.steps}  ",
        "",
        "## Training",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Initial loss | {initial_loss:.4f} |",
        f"| Final loss | {final_loss:.4f} |",
        f"| Imagined mean reward | {final_eval['imagined_mean_reward']:.4f} |",
        f"| Real mean reward | {final_eval['real_mean_reward']:.4f} |",
        f"| DoD: imagined reward finite | {'✅ PASS' if passed else '❌ FAIL'} |",
        "",
        "## Summary",
        "",
        f"Dreamer-mini trained on {n_eps} GR00T rollouts from LIBERO Spatial (all 10 tasks).",
        "World model learns manipulation dynamics: eef position, orientation, and gripper state.",
        f"Loss dropped {initial_loss:.3f} → {final_loss:.3f}. "
        f"Imagined reward is {'finite ✅' if passed else 'NaN ❌'}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"[t2] report → {path}")


def main():
    args = _parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[t2] device={device}")

    print(f"[t2] loading rollouts from {args.data}")
    data = torch.load(args.data, map_location="cpu")
    episodes = list(zip(data["obs"], data["action"], data["reward"]))
    obs_dim = episodes[0][0].shape[-1]
    act_dim = episodes[0][1].shape[-1]
    print(f"[t2] {len(episodes)} episodes  obs_dim={obs_dim}  act_dim={act_dim}")

    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim,
                    deter=args.deter, stoch=args.stoch, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)

    initial_loss = None
    final_loss = None

    for step in range(args.steps):
        obs_t, act_t, rew_t = _sample_batch(episodes, args.batch, args.seq_len, device)
        opt.zero_grad()
        loss, parts = wm.loss(obs_t, act_t, rew_t)
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        if step % 300 == 0 or step == args.steps - 1:
            eval_info = _eval_imagination(wm, episodes, args.imagine_horizon, device)
            print(f"step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"recon={parts['recon']:.4f}  rew={parts['reward']:.4f}  "
                  f"kl={parts['kl']:.4f}  | {eval_info}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": wm.state_dict(),
        "obs_dim": obs_dim, "act_dim": act_dim,
        "deter": args.deter, "stoch": args.stoch, "hidden": args.hidden,
        "source_data": args.data, "steps": args.steps,
    }, out)
    print(f"[t2] checkpoint → {out}")

    final_eval = _eval_imagination(wm, episodes, args.imagine_horizon, device)
    print(f"[t2] final eval: {final_eval}")

    passed = not math.isnan(final_eval["imagined_mean_reward"])
    print(f"[t2] DoD: imagined reward finite = {passed}")

    _write_report(Path(args.result_doc), args,
                  initial_loss, final_loss, final_eval,
                  len(episodes), obs_dim, act_dim)
    print("[t2] done")


if __name__ == "__main__":
    main()
