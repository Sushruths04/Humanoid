"""GR00T N1.7 policy wrapper for LIBERO evaluation.

Wraps the GR00T inference API (from isaac-gr00t) into the same interface used
by evaluate_manip.py, so T0 BC and T1 GR00T can be swapped in seamlessly.

Usage (inside evaluate_manip.py or standalone):
    from programs.t1_groot_lora.groot_policy import build_groot_policy
    policy_fn, compact = build_groot_policy(
        model_path="programs/checkpoints/groot_n17/libero_spatial/libero_spatial",
        embodiment_tag="LIBERO_PANDA",
        device="cuda",
    )
    # policy_fn(obs_dict) -> np.ndarray action (7,)
"""

from __future__ import annotations

import os
import sys

_GROOT_SRC = os.environ.get("GROOT_SRC", "/tmp/Isaac-GR00T")
if _GROOT_SRC not in sys.path:
    sys.path.insert(0, _GROOT_SRC)


def build_groot_policy(
    model_path: str,
    embodiment_tag: str = "LIBERO_PANDA",
    device: str = "cuda",
    action_horizon: int = 8,
    denoising_steps: int | None = None,
):
    """Return (policy_fn, compact=False).

    policy_fn(obs_dict: dict) -> np.ndarray of shape (7,)
    obs_dict keys expected (LIBERO env output):
        robot0_joint_pos (7,), robot0_eef_pos (3,), robot0_eef_quat (4,),
        robot0_gripper_qpos (2,), agentview_image (H, W, 3).
    """
    import numpy as np
    import torch
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    print(f"[gr00t] loading model from {model_path}")
    print(f"[gr00t] embodiment_tag={embodiment_tag}  device={device}")

    tag = EmbodimentTag(embodiment_tag.upper())
    policy = Gr00tPolicy.from_pretrained(
        model_path,
        embodiment_tag=tag,
        denoising_steps=denoising_steps,
    )
    policy.to(device)
    policy.eval()

    # Action horizon — GR00T outputs a chunk, we return the first action
    _action_horizon = action_horizon

    def policy_fn(obs_dict: dict) -> np.ndarray:
        """Map LIBERO env obs dict → 7-dim action."""
        obs = _libero_obs_to_groot(obs_dict, device)
        with torch.no_grad():
            action_chunk = policy.get_action(obs)  # (T, 7)
        # Return first action in chunk
        if hasattr(action_chunk, "cpu"):
            action_chunk = action_chunk.cpu().numpy()
        return np.asarray(action_chunk[0], dtype=np.float32)

    print("[gr00t] model loaded ✓")
    return policy_fn, False  # False = not compact (full env obs dict)


def _libero_obs_to_groot(obs_dict: dict, device: str) -> dict:
    """Convert LIBERO env obs dict to the format expected by Gr00tPolicy."""
    import numpy as np
    import torch

    def _t(x, dtype=torch.float32):
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype,
                            device=device).unsqueeze(0)

    # Robot state: joint pos + eef pos + eef quat + gripper
    joint = obs_dict.get("robot0_joint_pos", np.zeros(7))
    eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs_dict.get("robot0_eef_quat", np.zeros(4))
    gripper = obs_dict.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([joint, eef_pos, eef_quat, gripper])  # 16-dim

    # Image: (H, W, 3) uint8 → (1, 3, H, W) float32 normalized
    img = obs_dict.get("agentview_image", np.zeros((128, 128, 3), dtype=np.uint8))
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

    return {
        "state": _t(state),
        "image": img_t,
    }
