import torch
from programs.world_model.rssm import WorldModel


def test_loss_shapes_and_scalar():
    wm = WorldModel(obs_dim=5, action_dim=2, deter=32, stoch=8, hidden=32)
    obs = torch.randn(6, 4, 5)
    act = torch.randn(6, 4, 2)
    rew = torch.randn(6, 4)
    loss, parts = wm.loss(obs, act, rew)
    assert loss.dim() == 0
    assert set(parts) == {"recon", "reward", "kl"}


def test_imagine_shape():
    wm = WorldModel(obs_dim=5, action_dim=2, deter=32, stoch=8, hidden=32)
    state = wm.rssm.initial(4, "cpu")
    actor = lambda feat: torch.zeros(feat.shape[0], 2)
    feats = wm.imagine(state, actor, horizon=7)
    assert feats.shape == (7, 4, wm.feat_dim)


def test_world_model_overfits_tiny_batch():
    torch.manual_seed(0)
    wm = WorldModel(obs_dim=5, action_dim=2, deter=32, stoch=8, hidden=32)
    obs = torch.randn(6, 4, 5)
    act = torch.randn(6, 4, 2)
    rew = torch.randn(6, 4)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    first = float(wm.loss(obs, act, rew)[0])
    for _ in range(80):
        opt.zero_grad()
        loss, _ = wm.loss(obs, act, rew)
        loss.backward()
        opt.step()
    last = float(wm.loss(obs, act, rew)[0])
    assert last < first * 0.7   # the model learns to reconstruct/predict the batch


from programs.world_model.agent import Actor, imagine_returns


def test_actor_improves_imagined_reward():
    torch.manual_seed(0)
    wm = WorldModel(obs_dim=5, action_dim=2, deter=32, stoch=8, hidden=32)
    actor = Actor(wm.feat_dim, action_dim=2)
    state = wm.rssm.initial(8, "cpu")
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)

    def mean_imagined_reward():
        feats = wm.imagine(state, actor, horizon=5)
        return wm.reward_head(feats).mean()

    first = float(mean_imagined_reward().detach())
    for _ in range(60):
        opt.zero_grad()
        (-mean_imagined_reward()).backward()
        opt.step()
    last = float(mean_imagined_reward().detach())
    assert last > first   # actor learns actions that reach higher-reward latents
