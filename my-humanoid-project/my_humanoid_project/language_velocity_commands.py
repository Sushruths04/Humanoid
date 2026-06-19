"""Language-conditioning ON: velocity command set + target mapping + tracking reward.

This is the "turn language on" implementation (see ../../PLANS/LANGUAGE_ON_PLAN.md). It is
ADDITIVE — it does not modify the existing language_commands.py (which stays as the fixed
placeholder). The key difference vs the placeholder:
  - commands are RANDOMIZED per episode (a per-env id resampled on reset), and
  - a reward DEPENDS on the command (track the target base velocity it maps to),
so the policy actually has a reason and a way to condition on the embedding.

Pure PyTorch (+ reuses the existing deterministic hash embedding) → CPU-importable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .language_commands import LANGUAGE_EMBEDDING_DIM, embedding_for_text


@dataclass(frozen=True)
class VelCommand:
    cmd_id: int
    text: str
    vx: float       # target base linear x velocity (m/s)
    vy: float       # target base linear y velocity (m/s)
    yaw: float      # target base yaw rate (rad/s)


# Velocity-command set — the cleanest reward signal to prove language is "on".
VEL_COMMANDS = (
    VelCommand(0, "walk forward",   1.0, 0.0,  0.0),
    VelCommand(1, "walk backward", -1.0, 0.0,  0.0),
    VelCommand(2, "turn left",      0.0, 0.0,  0.5),
    VelCommand(3, "turn right",     0.0, 0.0, -0.5),
    VelCommand(4, "stand still",    0.0, 0.0,  0.0),
    VelCommand(5, "walk slow",      0.5, 0.0,  0.0),
    VelCommand(6, "walk fast",      1.5, 0.0,  0.0),
)
NUM_COMMANDS = len(VEL_COMMANDS)


def command_embeddings(device) -> torch.Tensor:
    """(NUM_COMMANDS, LANGUAGE_EMBEDDING_DIM) table of per-command hash embeddings."""
    rows = [embedding_for_text(c.text) for c in VEL_COMMANDS]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def command_targets(device) -> torch.Tensor:
    """(NUM_COMMANDS, 3) table of [vx, vy, yaw] targets per command."""
    rows = [[c.vx, c.vy, c.yaw] for c in VEL_COMMANDS]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def sample_command_ids(n: int, device) -> torch.Tensor:
    return torch.randint(0, NUM_COMMANDS, (n,), device=device)


def command_track_reward(base_lin_vel_b, base_ang_vel_b, cmd_ids, targets, sigma=0.5):
    """Gaussian tracking reward: match (vx, vy, yaw_rate) to the command's target.

    base_lin_vel_b: (N,3) base-frame linear velocity; base_ang_vel_b: (N,3) base-frame
    angular velocity; cmd_ids: (N,) long; targets: (NUM_COMMANDS,3). Returns (N,)."""
    tgt = targets[cmd_ids]                                   # (N,3)
    actual = torch.stack(
        [base_lin_vel_b[:, 0], base_lin_vel_b[:, 1], base_ang_vel_b[:, 2]], dim=-1
    )
    err2 = ((actual - tgt) ** 2).sum(dim=-1)
    return torch.exp(-err2 / (sigma ** 2))
