"""Dreamer-mini: a small recurrent state-space world model (P2), pure PyTorch.

Vector-observation world model with an RSSM latent (deterministic GRU state +
stochastic latent), decoder, and reward head. Learns dynamics from sequences of
(obs, action, reward) and can roll forward "in imagination" for model-based RL.
Kept small so it trains and tests on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter: int = 64, stoch: int = 16, hidden: int = 64):
        super().__init__()
        self.deter, self.stoch = deter, stoch
        self.pre_gru = nn.Sequential(nn.Linear(stoch + action_dim, hidden), nn.ELU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = nn.Sequential(nn.Linear(deter, hidden), nn.ELU(), nn.Linear(hidden, 2 * stoch))
        self.post_net = nn.Sequential(nn.Linear(deter + embed_dim, hidden), nn.ELU(), nn.Linear(hidden, 2 * stoch))

    def initial(self, batch: int, device) -> dict:
        return {"deter": torch.zeros(batch, self.deter, device=device),
                "stoch": torch.zeros(batch, self.stoch, device=device)}

    @staticmethod
    def _dist(params: torch.Tensor) -> torch.distributions.Normal:
        mean, std = params.chunk(2, dim=-1)
        return torch.distributions.Normal(mean, F.softplus(std) + 0.1)

    def img_step(self, state: dict, action: torch.Tensor) -> dict:
        x = self.pre_gru(torch.cat([state["stoch"], action], dim=-1))
        deter = self.gru(x, state["deter"])
        params = self.prior_net(deter)
        stoch = self._dist(params).rsample()
        return {"deter": deter, "stoch": stoch, "params": params}

    def obs_step(self, state: dict, action: torch.Tensor, embed: torch.Tensor):
        prior = self.img_step(state, action)
        params = self.post_net(torch.cat([prior["deter"], embed], dim=-1))
        stoch = self._dist(params).rsample()
        post = {"deter": prior["deter"], "stoch": stoch, "params": params}
        return post, prior

    @staticmethod
    def feature(state: dict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, deter: int = 64, stoch: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.rssm = RSSM(action_dim, embed_dim=hidden, deter=deter, stoch=stoch, hidden=hidden)
        feat = deter + stoch
        self.decoder = nn.Sequential(nn.Linear(feat, hidden), nn.ELU(), nn.Linear(hidden, obs_dim))
        self.reward_head = nn.Sequential(nn.Linear(feat, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.feat_dim = feat

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        # obs_seq (T, B, obs_dim); action_seq (T, B, action_dim)
        t_len, batch, _ = obs_seq.shape
        state = self.rssm.initial(batch, obs_seq.device)
        embeds = self.encoder(obs_seq)
        posts, priors = [], []
        for t in range(t_len):
            post, prior = self.rssm.obs_step(state, action_seq[t], embeds[t])
            posts.append(post)
            priors.append(prior)
            state = post
        return posts, priors

    def loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor, reward_seq: torch.Tensor, kl_scale: float = 1.0):
        posts, priors = self.observe(obs_seq, action_seq)
        feats = torch.stack([self.rssm.feature(p) for p in posts])
        recon_loss = F.mse_loss(self.decoder(feats), obs_seq)
        reward_loss = F.mse_loss(self.reward_head(feats).squeeze(-1), reward_seq)
        kl = torch.stack([
            torch.distributions.kl_divergence(self.rssm._dist(po["params"]), self.rssm._dist(pr["params"])).mean()
            for po, pr in zip(posts, priors)
        ]).mean()
        total = recon_loss + reward_loss + kl_scale * kl
        return total, {"recon": recon_loss.item(), "reward": reward_loss.item(), "kl": kl.item()}

    def imagine(self, init_state: dict, actor, horizon: int) -> torch.Tensor:
        state = {k: v for k, v in init_state.items() if k in ("deter", "stoch")}
        feats = []
        for _ in range(horizon):
            action = actor(self.rssm.feature(state))
            state = self.rssm.img_step(state, action)
            feats.append(self.rssm.feature(state))
        return torch.stack(feats)
