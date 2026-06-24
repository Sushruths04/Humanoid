"""Rope / pull force model (PLAN.md §3.3).

Pure PyTorch — no Isaac Lab dependency, so it is unit-testable on CPU.

Two models:
  - "spring": a virtual anchor moves forward at v_pull; a PD/spring force pulls the
    handle toward it, capped at f_max. Naturally caps speed ~ v_pull. (DEFAULT)
  - "force":  a constant horizontal force at the handle.

All quantities are batched: shape (num_envs, 3) for vectors, (num_envs,) for scalars.
"""
from __future__ import annotations

import torch

KMH_TO_MS = 1.0 / 3.6


def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh * KMH_TO_MS


class RopeModel:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        model: str = "spring",
        v_pull_kmh: float = 30.0,
        k_p: float = 800.0,
        k_d: float = 50.0,
        f_max: float = 600.0,
        pull_dir: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.model = model
        self.k_p = k_p
        self.k_d = k_d
        self.f_max = f_max
        self.pull_dir = torch.tensor(pull_dir, device=self.device, dtype=torch.float32)
        self.pull_dir = self.pull_dir / (self.pull_dir.norm() + 1e-8)

        # per-env target pull speed (m/s); can be set by the curriculum / DR
        self.v_pull = torch.full((num_envs,), kmh_to_ms(v_pull_kmh), device=self.device)
        # virtual anchor position (world), initialized at reset to handle + offset
        self.anchor_pos = torch.zeros((num_envs, 3), device=self.device)

    # ---- lifecycle ----
    def reset(self, env_ids: torch.Tensor, handle_pos: torch.Tensor, lead: float = 0.4):
        """Place the anchor `lead` metres ahead of the handle along the pull direction."""
        self.anchor_pos[env_ids] = handle_pos[env_ids] + lead * self.pull_dir

    def set_v_pull(self, v_pull_ms: torch.Tensor | float, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.v_pull[:] = v_pull_ms
        else:
            self.v_pull[env_ids] = v_pull_ms

    # ---- per-step ----
    def step_anchor(self, dt: float):
        """Advance the virtual anchor forward at v_pull."""
        self.anchor_pos += (self.v_pull.unsqueeze(-1) * dt) * self.pull_dir

    def compute_force(self, handle_pos: torch.Tensor, handle_vel: torch.Tensor) -> torch.Tensor:
        """Return the force (num_envs, 3) to apply at the handle body this step."""
        if self.model == "force":
            f = self.f_max * self.pull_dir.unsqueeze(0).expand(self.num_envs, 3)
            return f

        # spring / velocity-target model
        anchor_vel = self.v_pull.unsqueeze(-1) * self.pull_dir.unsqueeze(0)
        f = self.k_p * (self.anchor_pos - handle_pos) + self.k_d * (anchor_vel - handle_vel)
        # cap magnitude at f_max
        mag = f.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.f_max / (mag + 1e-8), max=1.0)
        return f * scale
