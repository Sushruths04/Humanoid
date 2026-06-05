"""CPU smoke tests for train_wm_isaac.py (uses synthetic rollout data)."""

import math
import random
import tempfile
from pathlib import Path

import torch
import pytest

from programs.world_model.rssm import WorldModel
from programs.world_model.train_wm_isaac import _sample_batch, _eval_imagination


def _synthetic_episodes(n_eps=20, obs_dim=4, act_dim=2, t_range=(10, 30)):
    eps = []
    for _ in range(n_eps):
        t = random.randint(*t_range)
        obs = torch.randn(t, obs_dim)
        act = torch.randn(t, act_dim)
        rew = -obs[:, :2].norm(dim=-1)  # point-mass style
        eps.append((obs, act, rew))
    return eps


def test_sample_batch_shape():
    eps = _synthetic_episodes()
    obs, act, rew = _sample_batch(eps, batch=8, seq_len=16, device="cpu")
    assert obs.shape == (16, 8, 4)
    assert act.shape == (16, 8, 2)
    assert rew.shape == (16, 8)


def test_sample_batch_pads_short_episodes():
    # episodes shorter than seq_len — must not crash
    eps = _synthetic_episodes(n_eps=5, t_range=(3, 5))
    obs, act, rew = _sample_batch(eps, batch=4, seq_len=16, device="cpu")
    assert obs.shape == (16, 4, 4)


def test_eval_imagination_returns_finite():
    eps = _synthetic_episodes(n_eps=20, obs_dim=4, act_dim=2)
    wm = WorldModel(obs_dim=4, action_dim=2, deter=32, stoch=8, hidden=32)
    info = _eval_imagination(wm, eps, horizon=5, device="cpu")
    assert not math.isnan(info["imagined_mean_reward"])
    assert not math.isnan(info["real_mean_reward"])


def test_train_loop_reduces_loss():
    """Loss should drop after a few gradient steps on synthetic data."""
    torch.manual_seed(42)
    eps = _synthetic_episodes(n_eps=50, obs_dim=4, act_dim=2, t_range=(20, 40))
    wm = WorldModel(obs_dim=4, action_dim=2, deter=32, stoch=8, hidden=32)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)

    losses = []
    for _ in range(30):
        obs, act, rew = _sample_batch(eps, batch=8, seq_len=10, device="cpu")
        opt.zero_grad()
        loss, _ = wm.loss(obs, act, rew)
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_save_load_rollout_file():
    """Verify the .pt rollout format written by collect_nav_rollouts can be read."""
    eps = _synthetic_episodes(n_eps=5)
    obs_list = [e[0] for e in eps]
    act_list = [e[1] for e in eps]
    rew_list = [e[2] for e in eps]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rollouts.pt"
        torch.save({
            "obs": obs_list,
            "action": act_list,
            "reward": rew_list,
            "task": "Humanoid-G1-CommandNav-v0",
            "checkpoint": "test",
            "num_episodes": len(eps),
        }, out)

        data = torch.load(out, map_location="cpu")
        assert data["num_episodes"] == 5
        assert len(data["obs"]) == 5
        assert data["obs"][0].shape[-1] == 4
