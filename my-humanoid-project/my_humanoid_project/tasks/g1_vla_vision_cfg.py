"""Vision-Language-Action (VLA) Unitree G1 environment configuration.

This task extends the marker navigation task with actual pixel observations 
from the G1's head camera.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm

from .g1_language_pickplace_cfg import LanguageConditionedG1CustomTaskCfg

@configclass
class G1VisionVLAEnvCfg(LanguageConditionedG1CustomTaskCfg):
    """G1 task with both language and vision observations."""

    def __post_init__(self):
        super().__post_init__()

        # 1. Add Camera to the Robot Scene
        # The G1 head usually has a camera around the 'head_link' or 'pelvis'
        # We attach a tiled camera to the head.
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
            update_period=0.1,  # 10Hz camera
            height=128,
            width=128,
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
        self.observations.policy.head_camera = ObsTerm(
            func=self._get_camera_rgb,
            params={"sensor_name": "camera"}
        )

    def _get_camera_rgb(self, env, sensor_name: str):
        """Extract RGB data from the specified camera sensor."""
        camera = env.scene[sensor_name]
        # Return flattened or processed RGB tensor
        return camera.data.output["rgb"]
