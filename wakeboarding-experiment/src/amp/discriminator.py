"""AMP discriminator + style reward (pure PyTorch, CPU-testable).

GAIL-style: a small MLP learns to tell reference (real wakeboard-stance) state-transitions
from policy ones. The style reward = -log(1 - D(s,s')) (clamped), per the AMP paper
(arxiv 2104.02180). Blend with task reward via the YAML `amp.reward_scale`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPDiscriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim * 2  # (s, s') transition
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, s, s_next):
        return self.net(torch.cat([s, s_next], dim=-1))

    def style_reward(self, s, s_next) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(s, s_next).squeeze(-1)
            prob = torch.sigmoid(logits)             # ~1 => looks like reference
            return torch.clamp(-torch.log(1.0 - prob + 1e-4), 0.0, 10.0)

    def loss(self, policy_s, policy_s_next, ref_s, ref_s_next, grad_pen_coef=10.0):
        """Standard AMP discriminator loss: LSGAN + gradient penalty on reference."""
        d_policy = self.forward(policy_s, policy_s_next)
        d_ref = self.forward(ref_s, ref_s_next)
        loss = F.mse_loss(d_policy, -torch.ones_like(d_policy)) + \
            F.mse_loss(d_ref, torch.ones_like(d_ref))
        # gradient penalty on reference samples
        ref_in = torch.cat([ref_s, ref_s_next], dim=-1).requires_grad_(True)
        d = self.net(ref_in)
        grad = torch.autograd.grad(d.sum(), ref_in, create_graph=True)[0]
        loss = loss + grad_pen_coef * (grad.norm(2, dim=-1) ** 2).mean()
        return loss
