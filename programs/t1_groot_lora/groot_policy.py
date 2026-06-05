"""GR00T N1.7 policy wrapper for LIBERO evaluation.

Wraps the isaac-gr00t Gr00tPolicy into the same step-callable interface used
by evaluate_groot_libero.py. Takes a LIBERO env obs dict and returns a 7-dim
action (OSC_POSE: x, y, z, roll, pitch, yaw, gripper).

Observation format expected by GR00T LIBERO_PANDA (libero_sim) model
— confirmed from processor_config.json in nvidia/GR00T-N1.7-LIBERO:

    video:
        image:        uint8  (1, 1, H, W, 3)  - agentview camera
        wrist_image:  uint8  (1, 1, H, W, 3)  - wrist camera
    state:  (per-joint-group — NOT a flat array)
        x:       float32 (1, 1, 1)
        y:       float32 (1, 1, 1)
        z:       float32 (1, 1, 1)
        roll:    float32 (1, 1, 1)
        pitch:   float32 (1, 1, 1)
        yaw:     float32 (1, 1, 1)
        gripper: float32 (1, 1, 2)  — both finger positions
    language:
        annotation.human.action.task_description: [["task text"]]

get_action() returns a dict with the same per-joint-group keys, each
(B, T_action, D). We assemble the first timestep into a 7-dim array.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_GROOT_SRC = os.environ.get("GROOT_SRC", "/tmp/Isaac-GR00T")
if _GROOT_SRC not in sys.path:
    sys.path.insert(0, _GROOT_SRC)

# Action key order matches modality.json and OSC_POSE convention
_ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] → [roll, pitch, yaw] in radians."""
    import scipy.spatial.transform as transform
    # LIBERO env returns [w, x, y, z], scipy expects [x, y, z, w]
    wxyz = quat.astype(np.float64)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return transform.Rotation.from_quat(xyzw).as_euler("xyz").astype(np.float32)


def load_groot_model(
    model_path: str,
    embodiment_tag: str = "LIBERO_PANDA",
    device: str = "cuda",
    denoising_steps: int | None = None,
):
    """Load GR00T model once; reuse across multiple tasks via make_policy_fn().

    Returns the raw Gr00tPolicy object — pass it to make_policy_fn() with a
    task-specific language string to get a callable for each task.
    """
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
    policy.model.eval()   # Gr00tPolicy wraps nn.Module as self.model
    print("[gr00t] model loaded ✓")
    return policy


def make_policy_fn(policy, task_language: str):
    """Wrap a loaded Gr00tPolicy with a fixed task language.

    Returns (policy_fn, compact) — same interface as build_groot_policy().
    policy_fn(obs_dict) → np.ndarray (7,)
    """
    _lang = task_language

    def policy_fn(obs_dict: dict) -> np.ndarray:
        obs = _libero_obs_to_groot(obs_dict, _lang)
        action_dict, _info = policy.get_action(obs)
        # action_dict keys: x, y, z, roll, pitch, yaw, gripper — each (B, T, D)
        return _action_dict_to_array(action_dict)

    return policy_fn, False  # False = use full obs_dict


def build_groot_policy(
    model_path: str,
    task_language: str,
    embodiment_tag: str = "LIBERO_PANDA",
    device: str = "cuda",
    action_horizon: int = 8,
    denoising_steps: int | None = None,
):
    """Build a callable GR00T policy for LIBERO evaluation (single-task).

    For multi-task eval, prefer load_groot_model() + make_policy_fn() to
    avoid reloading the 3B-param model for each task.

    Returns:
        (policy_fn, compact) — policy_fn(obs_dict) → np.ndarray (7,)
    """
    print(f"[gr00t] task: {task_language}")
    policy = load_groot_model(model_path, embodiment_tag, device, denoising_steps)
    return make_policy_fn(policy, task_language)


def _action_dict_to_array(action_dict: dict) -> np.ndarray:
    """Flatten per-joint-group action dict → 7-dim array (first timestep)."""
    parts = []
    for key in _ACTION_KEYS:
        val = np.asarray(action_dict[key], dtype=np.float32)  # (B, T, D)
        parts.append(val[0, 0].flatten())                      # (D,)
    return np.concatenate(parts)   # (7,)


def _libero_obs_to_groot(obs_dict: dict, task_language: str) -> dict:
    """Convert LIBERO env obs dict to the batched format expected by Gr00tPolicy.

    Observation shape convention: (B=1, T=1, ...)
    State is per-joint-group (not a flat array) — confirmed from
    processor_config.json: modality_keys=[x,y,z,roll,pitch,yaw,gripper].
    """
    eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(
        obs_dict.get("robot0_eef_quat", np.array([1., 0., 0., 0.])), dtype=np.float32
    )
    gripper = np.asarray(obs_dict.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)

    euler = _quat_to_euler(eef_quat)  # (3,) — roll, pitch, yaw

    def _s(v):
        """Scalar → (1, 1, 1) float32."""
        return np.array([[[float(v)]]], dtype=np.float32)

    # ── Images: (1, 1, H, W, 3) uint8 ────────────────────────────────────────
    def _prep_img(key, h=256, w=256):
        img = obs_dict.get(key, np.zeros((h, w, 3), dtype=np.uint8))
        return np.asarray(img, dtype=np.uint8)[None, None, ...]  # (1,1,H,W,3)

    return {
        "video": {
            "image":       _prep_img("agentview_image"),
            "wrist_image": _prep_img("robot0_eye_in_hand_image"),
        },
        "state": {
            "x":       _s(eef_pos[0]),
            "y":       _s(eef_pos[1]),
            "z":       _s(eef_pos[2]),
            "roll":    _s(euler[0]),
            "pitch":   _s(euler[1]),
            "yaw":     _s(euler[2]),
            "gripper": gripper[None, None, :],   # (1, 1, 2)
        },
        "language": {
            # Key from processor_config.json libero_sim section
            "annotation.human.action.task_description": [[task_language]],
        },
    }
