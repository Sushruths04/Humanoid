"""Reference-motion source for AMP (PLAN.md §9).

Option 1 (default, no external data): hand-authored keyframes (cannonball -> mid-rise ->
tall stance) interpolated into a short reference clip. Option 2 (video->pose retarget) drops
real clips into data/reference_motions/*.npz with the same schema.

Schema: each motion is an array (T, obs_dim) of the SAME observation features the AMP
discriminator sees (a subset of the policy obs — typically joint positions + base pose).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def build_keyframe_reference(obs_dim: int, fps: int = 50, seconds: float = 4.0) -> np.ndarray:
    """Interpolate 3 crude keyframes into a (T, obs_dim) clip. Replace indices with the
    real joint layout on GPU; this is a structural placeholder so AMP has something to fit."""
    T = int(fps * seconds)
    k0 = np.zeros(obs_dim, dtype=np.float32)   # cannonball: deep crouch (fill knee/hip flexed)
    k1 = np.zeros(obs_dim, dtype=np.float32)   # mid-rise
    k2 = np.zeros(obs_dim, dtype=np.float32)   # tall stance
    # TODO(on GPU): set k0/k1/k2 joint entries to crouch/mid/tall G1 poses.
    ts = np.linspace(0, 1, T)[:, None]
    half = T // 2
    a = k0[None] * (1 - ts[:half] * 2) + k1[None] * (ts[:half] * 2)
    b = k1[None] * (1 - (ts[half:] - 0.5) * 2) + k2[None] * ((ts[half:] - 0.5) * 2)
    return np.concatenate([a, b], axis=0).astype(np.float32)


class ReferenceMotionBuffer:
    def __init__(self, motions: list[np.ndarray], device):
        self.device = device
        # store as (sum_T-?, obs_dim) transitions
        self.s, self.s_next = [], []
        for m in motions:
            t = torch.as_tensor(m, device=device)
            self.s.append(t[:-1]); self.s_next.append(t[1:])
        self.s = torch.cat(self.s, 0); self.s_next = torch.cat(self.s_next, 0)

    @classmethod
    def from_dir(cls, path: str, obs_dim: int, device):
        p = Path(path)
        motions = []
        if p.exists():
            for f in sorted(p.glob("*.npz")):
                motions.append(np.load(f)["motion"])
        if not motions:
            motions = [build_keyframe_reference(obs_dim)]   # fallback keyframes
        return cls(motions, device)

    def sample(self, n: int):
        idx = torch.randint(0, self.s.shape[0], (n,), device=self.device)
        return self.s[idx], self.s_next[idx]
