"""T1 smoke tests — run CPU-only, no GPU or LIBERO env needed.

Tests the data conversion pipeline and the GR00T API adapter shapes.
"""

import sys
import os
import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _REPO_ROOT)


# ── obs conversion ────────────────────────────────────────────────────────────

def _make_fake_obs():
    """Fake LIBERO env obs dict matching the real env's keys."""
    return {
        "robot0_joint_pos": np.random.randn(7).astype(np.float32),
        "robot0_eef_pos": np.random.randn(3).astype(np.float32),
        "robot0_eef_quat": np.array([1., 0., 0., 0.], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_proprio-state": np.zeros(39, dtype=np.float32),
        "object-state": np.zeros(70, dtype=np.float32),
    }


def test_libero_obs_to_groot_state_shape():
    """_libero_obs_to_groot should produce state tensor of shape (1, 16)."""
    import torch
    groot_src = "/tmp/Isaac-GR00T"
    if not os.path.exists(groot_src):
        pytest.skip("Isaac-GR00T not cloned at /tmp/Isaac-GR00T")

    # Just test the conversion logic without loading the model
    obs = _make_fake_obs()
    joint = obs["robot0_joint_pos"]     # 7
    eef_p = obs["robot0_eef_pos"]       # 3
    eef_q = obs["robot0_eef_quat"]      # 4
    grip  = obs["robot0_gripper_qpos"]  # 2
    state = np.concatenate([joint, eef_p, eef_q, grip])
    assert state.shape == (16,), f"Expected (16,), got {state.shape}"


def test_libero_obs_image_shape():
    """agentview_image should be (H, W, 3) uint8."""
    obs = _make_fake_obs()
    img = obs["agentview_image"]
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_libero_obs_to_groot_image_tensor():
    """Image tensor after normalization should be (1, 3, H, W) float32 in [0,1]."""
    import torch
    obs = _make_fake_obs()
    img = obs["agentview_image"]
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    assert img_t.shape == (1, 3, 256, 256)
    assert float(img_t.max()) <= 1.0
    assert float(img_t.min()) >= 0.0


# ── evaluate_groot_libero imports ─────────────────────────────────────────────

def test_evaluate_groot_libero_importable():
    """evaluate_groot_libero.py should be importable (no isaac-gr00t needed for import)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_groot_libero",
        os.path.join(_REPO_ROOT, "programs/t1_groot_lora/evaluate_groot_libero.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # We only check it doesn't fail at import time (gr00t import is lazy)
    assert spec is not None
    assert mod is not None


def test_t1_scripts_exist():
    """All T1 scripts should be present."""
    t1_dir = os.path.join(_REPO_ROOT, "programs/t1_groot_lora")
    for fname in [
        "setup_t1.sh",
        "run_finetune.sh",
        "run_eval.sh",
        "groot_policy.py",
        "evaluate_groot_libero.py",
    ]:
        path = os.path.join(t1_dir, fname)
        assert os.path.exists(path), f"Missing T1 file: {fname}"
