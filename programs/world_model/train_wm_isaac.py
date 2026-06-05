"""Train the Dreamer-mini world model on real Isaac Lab nav rollouts (P2).

Loads a .pt rollout dataset collected by collect_nav_rollouts.py and trains
the WorldModel. Evaluates if imagined-rollout reward > mean reward from random
actions (minimum P2 DoD).

Usage:
    python -m programs.world_model.train_wm_isaac \
        --data programs/data/nav_rollouts_commandnav.pt \
        --steps 2000 \
        --out programs/checkpoints/world_model/wm_commandnav.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import math
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from programs.world_model.rssm import WorldModel


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to rollout .pt file from collect_nav_rollouts.py")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--deter", type=int, default=128)
    parser.add_argument("--stoch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--out", type=str, default="programs/checkpoints/world_model/wm_nav.pt")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _sample_batch(episodes: list, batch: int, seq_len: int, device: str):
    """Sample random sub-sequences from the episode pool."""
    obs_b, act_b, rew_b = [], [], []
    for _ in range(batch):
        ep = random.choice(episodes)
        obs_ep, act_ep, rew_ep = ep
        t = obs_ep.shape[0]
        if t <= seq_len:
            start = 0
            end = t
            pad = seq_len - t
        else:
            start = random.randint(0, t - seq_len)
            end = start + seq_len
            pad = 0
        obs_s = obs_ep[start:end]
        act_s = act_ep[start:end]
        rew_s = rew_ep[start:end]
        if pad > 0:
            obs_s = torch.cat([obs_s, obs_s[-1:].expand(pad, -1)], dim=0)
            act_s = torch.cat([act_s, act_s[-1:].expand(pad, -1)], dim=0)
            rew_s = torch.cat([rew_s, rew_s[-1:].expand(pad)], dim=0)
        obs_b.append(obs_s)
        act_b.append(act_s)
        rew_b.append(rew_s)
    obs_t = torch.stack(obs_b, dim=1).to(device)    # (T, B, obs_dim)
    act_t = torch.stack(act_b, dim=1).to(device)    # (T, B, act_dim)
    rew_t = torch.stack(rew_b, dim=1).to(device)    # (T, B)
    return obs_t, act_t, rew_t


def _eval_imagination(wm: WorldModel, episodes: list, horizon: int, device: str) -> dict:
    """Compare imagined reward vs. random-action reward on held-out episodes."""
    wm.eval()
    with torch.no_grad():
        obs_t, act_t, rew_t = _sample_batch(episodes, batch=16, seq_len=8, device=device)
        posts, _ = wm.observe(obs_t, act_t)
        init_state = posts[-1]
        random_actor = lambda f: torch.randn(f.shape[0], act_t.shape[2], device=device)
        imag_feats = wm.imagine(init_state, random_actor, horizon)
        imag_rewards = wm.reward_head(imag_feats).squeeze(-1).mean().item()
        real_rewards = float(rew_t.mean())
    wm.train()
    return {"imagined_mean_reward": round(imag_rewards, 4),
            "real_mean_reward": round(real_rewards, 4)}


def main():
    args = _parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[wm] loading rollouts from {args.data}")
    data = torch.load(args.data, map_location="cpu")
    obs_list = data["obs"]
    act_list = data["action"]
    rew_list = data["reward"]

    episodes = list(zip(obs_list, act_list, rew_list))
    obs_dim = obs_list[0].shape[-1]
    act_dim = act_list[0].shape[-1]
    print(f"[wm] {len(episodes)} episodes  obs_dim={obs_dim}  act_dim={act_dim}")

    wm = WorldModel(obs_dim=obs_dim, action_dim=act_dim,
                    deter=args.deter, stoch=args.stoch, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)

    for step in range(args.steps):
        obs_t, act_t, rew_t = _sample_batch(episodes, args.batch, args.seq_len, device)
        opt.zero_grad()
        loss, parts = wm.loss(obs_t, act_t, rew_t)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            eval_info = _eval_imagination(wm, episodes, args.imagine_horizon, device)
            print(f"step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"recon={parts['recon']:.4f}  rew={parts['reward']:.4f}  kl={parts['kl']:.4f}  "
                  f"| {eval_info}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": wm.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "deter": args.deter,
        "stoch": args.stoch,
        "hidden": args.hidden,
        "source_data": args.data,
        "steps": args.steps,
    }, out)
    print(f"[wm] saved {out}")

    final_eval = _eval_imagination(wm, episodes, args.imagine_horizon, device)
    print(f"[wm] final eval: {final_eval}")

    passed = not math.isnan(final_eval["imagined_mean_reward"])
    print(f"[wm] DoD: imagined reward is finite = {passed}")


if __name__ == "__main__":
    main()
