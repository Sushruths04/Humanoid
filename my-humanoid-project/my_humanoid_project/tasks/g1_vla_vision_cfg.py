"""Vision-Language-Action (VLA) Unitree G1 environment configuration.

This task extends the marker navigation task with actual pixel observations 
from the G1's head camera.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

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
    """Return normalized CHW RGB observations from the camera sensor."""

    camera = env.scene[sensor_name]
    rgb = camera.data.output["rgb"].float() / 255.0
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    if not getattr(get_camera_rgb, "_shape_logged", False):
        print(f"[VLA] Camera RGB observation shape: {tuple(rgb.shape)}")
        get_camera_rgb._shape_logged = True
    return rgb


@configclass
class G1VisionVLAImageObsCfg(ObsGroup):
    """Separate image observation group for the CNN policy encoder."""

    head_camera = ObsTerm(func=get_camera_rgb, params={"sensor_name": "camera"})

    def __post_init__(self):
        self.enable_corruption = False


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

        height = _env_int("VLA_CAMERA_HEIGHT", 128)
        width = _env_int("VLA_CAMERA_WIDTH", 128)
        update_period = _env_float("VLA_CAMERA_UPDATE_PERIOD", 0.05)
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
            self.observations.images = G1VisionVLAImageObsCfg()
        else:
            print("[VLA] Camera observation disabled by VLA_ENABLE_CAMERA_OBS=0.")


@configclass
class G1VisionVLACnnRunnerCfg(RslRlOnPolicyRunnerCfg):
    """CNN-based runner for the Vision-VLA task."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_vla_vision_cnn"
    obs_groups = {"actor": ["policy", "images"], "critic": ["policy", "images"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="relu",
            padding="none",
            norm="none",
            max_pool=False,
            global_pool="none",
            flatten=True,
        ),
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="relu",
            padding="none",
            norm="none",
            max_pool=False,
            global_pool="none",
            flatten=True,
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = int(_env_int("VISION_VLA_MAX_ITERS", self.max_iterations))
        self.experiment_name = "g1_vla_vision_cnn"
        self.algorithm.share_cnn_encoders = True


G1FlatPPORunnerCfg = G1VisionVLACnnRunnerCfg
