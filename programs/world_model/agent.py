"""Imagination actor-critic for the Dreamer-mini world model (P2 CP2.3).

The actor and critic are trained on latent trajectories rolled out *in the world
models imagination* (no simulator in the loop) — the core model-based-RL payoff.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(), nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


def imagine_returns(world_model, actor, init_state, horizon: int, gamma: float = 0.99):
    """Roll the actor out in imagination; return discounted returns and feats."""
    feats = world_model.imagine(init_state, actor, horizon)       # (H, B, feat)
    rewards = world_model.reward_head(feats).squeeze(-1)          # (H, B)
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(horizon)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns, feats
