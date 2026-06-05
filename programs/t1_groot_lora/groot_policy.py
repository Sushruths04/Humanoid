"""GR00T N1.7 policy wrapper for LIBERO evaluation.

Wraps the isaac-gr00t Gr00tPolicy into the same step-callable interface used
by evaluate_groot_libero.py. Takes a LIBERO env obs dict and returns a 7-dim
action (OSC_POSE delta: dx, dy, dz, droll, dpitch, dyaw, gripper).

Observation format expected by GR00T LIBERO_PANDA model:
    video:
        image:        uint8  (1, 1, H, W, 3)  - agentview camera
        wrist_image:  uint8  (1, 1, H, W, 3)  - wrist camera
    state:
        state:        float32 (1, 1, 8)        - eef_xyz + euler_rpy + gripper
    language:
        human.action.task_description: [["task text"]]

The LIBERO env provides all of these via obs_dict keys.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

_GROOT_SRC = os.environ.get("GROOT_SRC", "/tmp/Isaac-GR00T")
if _GROOT_SRC not in sys.path:
    sys.path.insert(0, _GROOT_SRC)


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] → [roll, pitch, yaw] in radians."""
    import scipy.spatial.transform as transform
    # LIBERO env returns [w, x, y, z], scipy expects [x, y, z, w]
    wxyz = quat.astype(np.float64)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    euler = transform.Rotation.from_quat(xyzw).as_euler("xyz")
    return euler.astype(np.float32)


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
        acts = action_dict["actions"]
        if hasattr(acts, "cpu"):
            acts = acts.cpu().numpy()
        acts = np.asarray(acts, dtype=np.float32)
        return acts[0, 0]  # first action in the chunk

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

    For multi-task eval (different language per task), prefer load_groot_model()
    + make_policy_fn() to avoid reloading the 3B-param model for each task.

    Returns:
        (policy_fn, compact) — policy_fn(obs_dict) → np.ndarray (7,)
    """
    print(f"[gr00t] task: {task_language}")
    policy = load_groot_model(model_path, embodiment_tag, device, denoising_steps)
    return make_policy_fn(policy, task_language)


def _libero_obs_to_groot(obs_dict: dict, task_language: str) -> dict:
    """Convert LIBERO env obs dict to the batched format expected by Gr00tPolicy.

    Observation shape convention: (B=1, T=1, ...)
    """
    # ── State: eef_xyz + euler_rpy + gripper_qpos = 8 dims ───────────────────
    eef_pos = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(obs_dict.get("robot0_eef_quat", np.array([1., 0., 0., 0.])), dtype=np.float32)
    gripper = np.asarray(obs_dict.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)

    euler = _quat_to_euler(eef_quat)                      # (3,)
    state = np.concatenate([eef_pos, euler, gripper])     # (8,)
    state_batched = state[None, None, :]                   # (1, 1, 8)

    # ── Images: (1, 1, H, W, 3) uint8 ────────────────────────────────────────
    def _prep_img(key, h=256, w=256):
        img = obs_dict.get(key, np.zeros((h, w, 3), dtype=np.uint8))
        img = np.asarray(img, dtype=np.uint8)
        return img[None, None, ...]                         # (1, 1, H, W, 3)

    img_main = _prep_img("agentview_image")
    img_wrist = _prep_img("robot0_eye_in_hand_image")

    # ── Language ──────────────────────────────────────────────────────────────
    # Shape: (B=1, T=1) — each element is a string
    lang = [[task_language]]

    return {
        "video": {
            "image": img_main,           # (1,1,H,W,3) uint8
            "wrist_image": img_wrist,    # (1,1,H,W,3) uint8
        },
        "state": {
            "state": state_batched,      # (1,1,8) float32
        },
        "language": {
            "human.action.task_description": lang,
        },
    }
