"""Vision-Language-Action (VLA) Unitree G1 environment configuration.

This task extends the marker navigation task with actual pixel observations 
from the G1's head camera.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm

from .g1_language_pickplace_cfg import LanguageConditionedG1CustomTaskCfg


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def get_camera_rgb(env, sensor_name: str):
    """Return flattened normalized RGB observations from the camera sensor."""

    camera = env.scene[sensor_name]
    rgb = camera.data.output["rgb"].float() / 255.0
    if not getattr(get_camera_rgb, "_shape_logged", False):
        print(f"[VLA] Camera RGB observation shape: {tuple(rgb.shape)}")
        get_camera_rgb._shape_logged = True
    return rgb.reshape(rgb.shape[0], -1).contiguous()


@configclass
class G1VisionVLAEnvCfg(LanguageConditionedG1CustomTaskCfg):
    """G1 task with both language and vision observations."""

    def __post_init__(self):
        super().__post_init__()

        # 1. Add Camera to the Robot Scene
        # The G1 head usually has a camera around the 'head_link' or 'pelvis'
        # We attach a tiled camera to the head.
        if not _env_bool("VLA_ENABLE_CAMERA", True):
            print("[VLA] Camera sensor disabled by VLA_ENABLE_CAMERA=0.")
            return

        height = _env_int("VLA_CAMERA_HEIGHT", 64)
        width = _env_int("VLA_CAMERA_WIDTH", 64)
        update_period = _env_float("VLA_CAMERA_UPDATE_PERIOD", 0.2)
        print(f"[VLA] Configuring tiled camera: {width}x{height}, update_period={update_period}s")

        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
            update_period=update_period,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.1, 0.0, 0.1), 
                rot=(1.0, 0.0, 0.0, 0.0), 
                convention="ros"
            ),
        )

        # 2. Add Vision to Policy Observations
        # We add the RGB data as a new observation group
        if _env_bool("VLA_ENABLE_CAMERA_OBS", True):
            self.observations.policy.head_camera = ObsTerm(
                func=get_camera_rgb,
                params={"sensor_name": "camera"},
            )
        else:
            print("[VLA] Camera observation disabled by VLA_ENABLE_CAMERA_OBS=0.")
