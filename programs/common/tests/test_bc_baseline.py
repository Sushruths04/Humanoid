"""CPU smoke tests for the MLPBCPolicy and training loop (no LIBERO needed)."""

import torch
import pytest
from programs.t0_manip_foundation.bc_baseline import MLPBCPolicy


def test_mlp_forward_shape():
    policy = MLPBCPolicy(obs_dim=10, action_dim=4)
    obs = torch.randn(8, 10)  # batch of 8
    out = policy(obs)
    assert out.shape == (8, 4)


def test_mlp_forward_deterministic():
    policy = MLPBCPolicy(obs_dim=6, action_dim=3)
    obs = torch.randn(4, 6)
    out1 = policy(obs)
    out2 = policy(obs)
    assert torch.allclose(out1, out2)


def test_mlp_has_parameters():
    policy = MLPBCPolicy(obs_dim=8, action_dim=2)
    params = list(policy.parameters())
    assert len(params) > 0


def test_mlp_gradient_flows():
    policy = MLPBCPolicy(obs_dim=4, action_dim=2)
    obs = torch.randn(2, 4)
    target = torch.randn(2, 2)
    loss = torch.nn.functional.mse_loss(policy(obs), target)
    loss.backward()
    for p in policy.parameters():
        assert p.grad is not None


def test_mlp_to_device_cpu():
    policy = MLPBCPolicy(obs_dim=4, action_dim=2)
    policy = policy.to("cpu")
    obs = torch.randn(3, 4)
    out = policy(obs)
    assert out.shape == (3, 2)


def test_mlp_state_dict_saveable():
    import io
    policy = MLPBCPolicy(obs_dim=4, action_dim=2)
    buf = io.BytesIO()
    torch.save(policy.state_dict(), buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    assert isinstance(loaded, dict)
    assert len(loaded) > 0
