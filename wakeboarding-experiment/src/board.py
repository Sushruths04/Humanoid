"""Wakeboard rigid-body asset + foot binding (PLAN.md §3.2).

Isaac Lab asset config. CPU-import-guarded so the module loads without Isaac Sim.
The board is a thin box welded to the G1 feet; it slides on a frictional ground that
approximates sand (μ ~ 0.3-0.5).

VERIFY ON GPU: exact RigidObjectCfg / fixed-joint API and prim paths against the
installed Isaac Lab version; foot binding may instead be done via an articulation
fixed joint in the robot USD.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except Exception:  # CPU fallback
    sim_utils = None
    RigidObjectCfg = object
    def configclass(cls):  # type: ignore
        return cls
    ISAACLAB_AVAILABLE = False


@dataclass
class BoardParams:
    length: float = 1.4
    width: float = 0.4
    thickness: float = 0.04
    mass: float = 3.0
    sand_friction: float = 0.4


def make_board_cfg(p: BoardParams):
    """Return a RigidObjectCfg for the wakeboard, or None on CPU."""
    if not ISAACLAB_AVAILABLE:
        return None
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        spawn=sim_utils.CuboidCfg(
            size=(p.length, p.width, p.thickness),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=p.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=p.sand_friction,
                dynamic_friction=p.sand_friction,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.7, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )


# Foot->board binding: implemented at env-build time. Preferred approach is a fixed
# joint per foot. See wakeboard_start_cfg.py::_bind_feet_to_board for the wiring point.
# Resolved from IsaacLab G1_CFG (g1.usd, 23-DoF): the foot bodies are the ankle-roll links.
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
