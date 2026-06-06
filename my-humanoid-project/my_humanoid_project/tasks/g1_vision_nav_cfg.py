"""P3 — Vision Navigation: camera-conditioned CommandNav for G1.

Extends P0 CommandNav with a head-mounted TiledCamera.
The CNN policy sees 84×84 RGB alongside command/proprioceptive features.
DoD: ≥60% success on CommandNav with pixel observations.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .g1_command_nav_cfg import CommandConditionedG1NavCfg

VISION_NAV_TASK_ID = "Humanoid-G1-VisionNav-v0"

_CAM_H = int(os.environ.get("P3_CAM_H", "84"))
_CAM_W = int(os.environ.get("P3_CAM_W", "84"))
_MAX_ITERS = int(os.environ.get("P3_MAX_ITERS", "1500"))
_NUM_STEPS = int(os.environ.get("P3_NUM_STEPS", "48"))


def _get_camera_rgb(env, sensor_name: str = "head_camera"):
    """Return normalized CHW RGB from the named camera sensor."""
    cam = env.scene[sensor_name]
    rgb = cam.data.output["rgb"].float() / 255.0
    return rgb.permute(0, 3, 1, 2).contiguous()


@configclass
class VisionNavImageObsCfg(ObsGroup):
    """Image observation group — RGB from head camera, no noise."""

    head_camera = ObsTerm(func=_get_camera_rgb, params={"sensor_name": "head_camera"})

    def __post_init__(self):
        self.enable_corruption = False


@configclass
class G1VisionNavEnvCfg(CommandConditionedG1NavCfg):
    """P3: CommandNav + head TiledCamera for pixel-conditioned nav."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.head_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
            update_period=0.05,
            height=_CAM_H,
            width=_CAM_W,
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
                convention="ros",
            ),
        )
        self.observations.images = VisionNavImageObsCfg()


@configclass
class G1VisionNavCnnRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO + CNN runner for P3 VisionNav.

    Tuned for A100 80GB: large batches, shared encoders, adaptive LR.
    """

    num_steps_per_env = _NUM_STEPS
    max_iterations = _MAX_ITERS
    save_interval = 100
    experiment_name = "g1_vision_nav"
    obs_groups = {"actor": ["policy", "images"], "critic": ["policy", "images"]}

    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
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
        hidden_dims=[512, 256, 128],
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
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.share_cnn_encoders = True
        self.experiment_name = "g1_vision_nav"
