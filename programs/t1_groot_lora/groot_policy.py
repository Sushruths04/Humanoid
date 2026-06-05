"""GR00T N1.7 policy wrapper for LIBERO evaluation.

Wraps isaac-gr00t's Gr00tPolicy into the same step-callable interface used
by evaluate_groot_libero.py. Takes a LIBERO env obs dict and returns a 7-dim
action ready for env.step().

Observation format — confirmed from official libero_env.py wrapper and
processor_config.json in nvidia/GR00T-N1.7-LIBERO:

    video:
        image:        uint8  (1,1,H,W,3)  - agentview camera, FLIPPED both axes
        wrist_image:  uint8  (1,1,H,W,3)  - wrist camera, FLIPPED both axes
    state:  (per-joint-group — NOT a flat array)
        x, y, z:     float32 (1,1,1)  — raw eef xyz
        roll, pitch, yaw: float32 (1,1,1) — quat2axisangle (NOT Euler)
        gripper:     float32 (1,1,2)  — robot0_gripper_qpos both fingers
    language:
        annotation.human.action.task_description: [["task text"]]

Action post-processing (matches libero_env.py step()):
    gripper = np.sign(2*gripper - 1)   # [0,1] → binarized [-1,+1]
    gripper = -gripper                  # invert: dataset 1=open → LIBERO -1=open
"""

from __future__ import annotations

import os
import sys

import numpy as np

_GROOT_SRC = os.environ.get("GROOT_SRC", "/tmp/Isaac-GR00T")
if _GROOT_SRC not in sys.path:
    sys.path.insert(0, _GROOT_SRC)

_ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle (matches robosuite quat2axisangle).

    Uses the same implementation as libero_env.py so state matches training.
    """
    import math
    quat = np.asarray(quat, dtype=np.float64)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def load_groot_model(
    model_path: str,
    embodiment_tag: str = "LIBERO_PANDA",
    device: str = "cuda",
    denoising_steps: int | None = None,
):
    """Load GR00T model once; reuse across multiple tasks via make_policy_fn()."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    kwargs = {}
    if denoising_steps is not None:
        kwargs["denoising_steps"] = denoising_steps

    print(f"[gr00t] loading from {model_path} (device={device})")
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        strict=False,
        **kwargs,
    )
    policy.model.eval()
    print("[gr00t] model loaded ✓")
    return policy


def make_policy_fn(policy, task_language: str):
    """Wrap a loaded Gr00tPolicy with a fixed task language.

    Returns (policy_fn, compact):
        policy_fn(obs_dict) → np.ndarray (7,) ready for env.step()
    """
    _lang = task_language

    def policy_fn(obs_dict: dict) -> np.ndarray:
        obs = _libero_obs_to_groot(obs_dict, _lang)
        action_dict, _info = policy.get_action(obs)
        action = _action_dict_to_array(action_dict)
        return _apply_gripper_transform(action)

    return policy_fn, False


def build_groot_policy(
    model_path: str,
    task_language: str,
    embodiment_tag: str = "LIBERO_PANDA",
    device: str = "cuda",
    action_horizon: int = 8,
    denoising_steps: int | None = None,
):
    """Build a callable GR00T policy for single-task LIBERO evaluation."""
    print(f"[gr00t] task: {task_language}")
    policy = load_groot_model(model_path, embodiment_tag, device, denoising_steps)
    return make_policy_fn(policy, task_language)


# ── helpers ───────────────────────────────────────────────────────────────────

def _action_dict_to_array(action_dict: dict) -> np.ndarray:
    """Flatten per-joint-group action dict → 7-dim array (first timestep)."""
    parts = []
    for key in _ACTION_KEYS:
        val = np.asarray(action_dict[key], dtype=np.float32)  # (B, T, D)
        parts.append(val[0, 0].flatten())
    return np.concatenate(parts)  # (7,)


def _apply_gripper_transform(action: np.ndarray) -> np.ndarray:
    """Convert GR00T gripper output to LIBERO env.step() convention.

    GR00T (LeRobot dataset): 0 = close, 1 = open
    LIBERO OSC_POSE:         +1 = close, -1 = open

    Matches libero_env.py: normalize_gripper_action(binarize=True) then invert.
    """
    a = action.copy()
    a[-1] = -np.sign(2.0 * a[-1] - 1.0)   # [0,1]→[-1,+1]→binarize→invert
    return a


def _libero_obs_to_groot(obs_dict: dict, task_language: str) -> dict:
    """Convert LIBERO env obs dict → batched GR00T input dict.

    Matches _process_observation() in official libero_env.py:
      - Images are flipped on both axes ([::-1, ::-1])
      - Rotation uses quat2axisangle (not Euler)
      - State is per-joint-group (not flat)
    """
    eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(
        obs_dict.get("robot0_eef_quat", np.array([0., 0., 0., 1.])), dtype=np.float32
    )
    gripper = np.asarray(obs_dict.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)

    rpy = _quat2axisangle(eef_quat)  # axis-angle, (3,)

    def _s(v):
        return np.array([[[float(v)]]], dtype=np.float32)  # (1,1,1)

    def _prep_img(key, h=256, w=256):
        img = obs_dict.get(key, np.zeros((h, w, 3), dtype=np.uint8))
        img = np.asarray(img, dtype=np.uint8)[::-1, ::-1]   # flip both axes
        return img[None, None, ...]  # (1,1,H,W,3)

    return {
        "video": {
            "image":       _prep_img("agentview_image"),
            "wrist_image": _prep_img("robot0_eye_in_hand_image"),
        },
        "state": {
            "x":       _s(eef_pos[0]),
            "y":       _s(eef_pos[1]),
            "z":       _s(eef_pos[2]),
            "roll":    _s(rpy[0]),
            "pitch":   _s(rpy[1]),
            "yaw":     _s(rpy[2]),
            "gripper": gripper[None, None, :],  # (1,1,2)
        },
        "language": {
            "annotation.human.action.task_description": [[task_language]],
        },
    }
