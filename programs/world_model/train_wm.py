"""Toy training script for the Dreamer-mini world model (P2).

Generates rollouts from a simple 2D point-mass (obs = [pos, vel], action =
acceleration, reward = -distance to origin), trains the world model, and prints
the loss curve. Pure CPU; a self-contained "does it learn dynamics" demo.

Run:  python programs/world_model/train_wm.py --steps 300
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from programs.world_model.rssm import WorldModel


def rollout_pointmass(batch, length, dt=0.1):
    pos = torch.randn(batch, 2)
    vel = torch.zeros(batch, 2)
    obs_seq, act_seq, rew_seq = [], [], []
    for _ in range(length):
        action = torch.randn(batch, 2) * 0.5
        vel = vel + action * dt
        pos = pos + vel * dt
        obs_seq.append(torch.cat([pos, vel], dim=-1))
        act_seq.append(action)
        rew_seq.append(-pos.norm(dim=-1))
    return torch.stack(obs_seq), torch.stack(act_seq), torch.stack(rew_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args()

    wm = WorldModel(obs_dim=4, action_dim=2, deter=64, stoch=16, hidden=64)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    for step in range(args.steps):
        obs, act, rew = rollout_pointmass(args.batch, args.length)
        opt.zero_grad()
        loss, parts = wm.loss(obs, act, rew)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print("step", step, "loss", round(float(loss), 4),
                  "recon", round(parts["recon"], 4),
                  "reward", round(parts["reward"], 4),
                  "kl", round(parts["kl"], 4))


if __name__ == "__main__":
    main()
