"""Wakeboard reward terms (PLAN.md §5).

Each function follows the Isaac Lab manager-based reward signature:
    func(env, **params) -> torch.Tensor of shape (num_envs,)

These are written to be wired into a RewardManager via RewTerm(func=..., weight=...).
The state accessors (env.scene["robot"], board pitch, handle pose, rope force) assume
helper buffers populated by the env in wakeboard_start_cfg.py. Spots that depend on the
exact Isaac Lab API are marked `# VERIFY`.

Design notes:
- Angles in radians internally; thresholds documented in degrees.
- All terms return UNWEIGHTED rewards; weights live in the YAML config.
"""
from __future__ import annotations

import math

import torch

DEG = math.pi / 180.0


# ---------------------------------------------------------------- helpers
def _robot(env):
    return env.scene["robot"]  # VERIFY asset key


def _gravity_up_proj(env) -> torch.Tensor:
    """Torso uprightness in [-1,1]: projection of -gravity onto torso z-axis (1=vertical)."""
    # VERIFY: Isaac Lab exposes projected_gravity_b (gravity in base frame).
    proj_g = _robot(env).data.projected_gravity_b  # (N,3), points "down" in base frame
    return -proj_g[:, 2]


def _pelvis_height(env) -> torch.Tensor:
    return _robot(env).data.root_pos_w[:, 2]  # VERIFY: pelvis/root height


# ---------------------------------------------------------------- task rewards (§5.1)
def pelvis_height(env, h_target: float = 0.6) -> torch.Tensor:
    return torch.clamp(_pelvis_height(env), 0.0, h_target) / h_target


def height_progress(env, phase_gate: float = 0.5) -> torch.Tensor:
    """Positive delta pelvis height, gated to only count after phase_gate seconds
    (discourages standing up too fast — wakeboard rule #1)."""
    h = _pelvis_height(env)
    prev = getattr(env, "_prev_pelvis_h", h)
    dh = torch.clamp(h - prev, min=0.0)
    env._prev_pelvis_h = h.detach()
    gate = (env.episode_length_buf * env.step_dt >= phase_gate).float()  # VERIFY buf name
    return dh * gate


def uprightness(env) -> torch.Tensor:
    return torch.clamp(_gravity_up_proj(env), 0.0, 1.0)


def survival(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def forward_glide(env, sigma: float = 1.0) -> torch.Tensor:
    """Reward board forward speed tracking the current pull target v_pull (gaussian)."""
    board_vx = env._board_lin_vel[:, 0]            # populated by env
    target = env.rope.v_pull                       # (N,) m/s
    return torch.exp(-((board_vx - target) ** 2) / (sigma ** 2))


def success_bonus(env) -> torch.Tensor:
    """Sparse +1 (weighted to +50) the step a held stable ride is first achieved."""
    return env._success_event.float()              # env sets this when §2 criteria hold >=1.5s


# ---------------------------------------------------------------- biomechanics (§5.2)
def board_positive_angle(env, lo_deg: float = 10.0, hi_deg: float = 20.0) -> torch.Tensor:
    """Reward board pitch in [lo,hi] deg, tapering outside (wakeboard rule #2)."""
    pitch = env._board_pitch                       # radians, +=nose up
    lo, hi = lo_deg * DEG, hi_deg * DEG
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    # 1.0 inside band, gaussian taper outside
    inside = ((pitch >= lo) & (pitch <= hi)).float()
    taper = torch.exp(-((pitch - center) ** 2) / (2 * (half ** 2) + 1e-6))
    return torch.maximum(inside, taper)


def arms_straight(env) -> torch.Tensor:
    """Reward elbow extension (rule #3): 1 when elbows near full extension, 0 when flexed."""
    elbow = env._elbow_flexion                     # (N,) radians, 0=straight
    return torch.exp(-(elbow ** 2) / (2 * (20 * DEG) ** 2))


def handle_at_hips(env) -> torch.Tensor:
    """Negative distance of handle from a hip-height target near the pelvis (rule #3)."""
    d = (env._handle_pos - env._hip_target_pos).norm(dim=-1)
    return torch.exp(-(d ** 2) / (2 * (0.15 ** 2)))


def lean_back_moderate(env, lo_deg: float = 10.0, hi_deg: float = 25.0) -> torch.Tensor:
    """Reward torso back-lean in band (rule #5); penalize if back-lean co-occurs with
    elbow flexion (= pulling against the rope, rule #4)."""
    lean = env._torso_back_lean                    # radians
    lo, hi = lo_deg * DEG, hi_deg * DEG
    band = ((lean >= lo) & (lean <= hi)).float()
    pulling = (env._elbow_flexion > 15 * DEG).float()
    return band * (1.0 - 0.5 * pulling)


def knee_bend_maintained(env, min_deg: float = 20.0) -> torch.Tensor:
    """Reward keeping knees bent early, relaxing the requirement as phase->1 (rules #1,#5)."""
    knee = env._knee_flexion                       # (N,) radians, larger=more bent
    phase = torch.clamp(env.episode_length_buf * env.step_dt / env._t_success, 0.0, 1.0)
    required = (min_deg * DEG) * (1.0 - phase)     # required bend decreases over time
    return (knee >= required).float()


# ---------------------------------------------------------------- penalties (§5.4)
def pen_stand_too_fast(env, vmax: float = 0.6, early_phase: float = 0.4) -> torch.Tensor:
    """Penalize high pelvis vertical velocity early in the episode (rule #1)."""
    vz = _robot(env).data.root_lin_vel_w[:, 2]
    early = (env.episode_length_buf * env.step_dt / env._t_success < early_phase).float()
    return early * torch.clamp(vz - vmax, min=0.0)


def pen_pull_against_rope(env) -> torch.Tensor:
    """Penalize elbow-flexion effort while the rope force is high (rule #4)."""
    rope_mag = env._rope_force.norm(dim=-1)
    flex = torch.clamp(env._elbow_flexion, min=0.0)
    return (rope_mag / (env.rope.f_max + 1e-6)) * flex


def pen_torque(env) -> torch.Tensor:
    return torch.sum(_robot(env).data.applied_torque ** 2, dim=-1)  # VERIFY attr


def pen_action_rate(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1)


def pen_action_accel(env) -> torch.Tensor:
    a = env.action_manager.action
    p = env.action_manager.prev_action
    pp = getattr(env, "_prev_prev_action", p)
    env._prev_prev_action = p.detach()
    return torch.sum((a - 2 * p + pp) ** 2, dim=-1)


def pen_dof_pos_limits(env) -> torch.Tensor:
    # VERIFY: Isaac Lab provides a soft dof-limit helper; placeholder L2-over-violation here.
    j = _robot(env).data.joint_pos
    lo = _robot(env).data.soft_joint_pos_limits[..., 0]
    hi = _robot(env).data.soft_joint_pos_limits[..., 1]
    below = torch.clamp(lo - j, min=0.0)
    above = torch.clamp(j - hi, min=0.0)
    return torch.sum(below + above, dim=-1)


def pen_fall(env) -> torch.Tensor:
    """Terminal penalty handled via termination; expose as a reward hook if desired."""
    return env._fall_event.float()
