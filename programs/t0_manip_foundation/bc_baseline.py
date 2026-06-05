"""Behaviour-cloning baseline for T0 manipulation (CPT0.3).

Trains a simple MLP or ACT policy via imitation on a LeRobot dataset.
CPU-runnable for smoke tests; GPU needed for meaningful performance.

Usage (LeRobot dataset from HuggingFace):
    python -m programs.t0_manip_foundation.bc_baseline \
        --dataset lerobot/aloha_sim_insertion_human \
        --task libero_spatial \
        --epochs 100 \
        --batch-size 32 \
        --out programs/checkpoints/t0_bc/policy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Behaviour-cloning baseline.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace LeRobot dataset repo id")
    parser.add_argument("--task", type=str, default="libero_spatial")
    parser.add_argument("--policy-type", type=str, default="mlp",
                        choices=["mlp", "act"],
                        help="mlp = simple MLP BC; act = Action Chunking Transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="programs/checkpoints/t0_bc/policy.pt")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class MLPBCPolicy:
    """Minimal MLP behaviour-cloning policy for smoke-test purposes.

    Obs -> hidden -> hidden -> action (MSE loss on demonstrated actions).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        import torch
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def __call__(self, obs):
        return self.forward(obs)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def to(self, device):
        self.net = self.net.to(device)
        return self


def train_bc(args) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = _load_lerobot_dataset(args.dataset, args.task)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    obs_dim, action_dim = _infer_dims(dataset)
    policy = MLPBCPolicy(obs_dim, action_dim).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"[bc] dataset={args.dataset} task={args.task} obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[bc] epochs={args.epochs} batch={args.batch_size} device={device}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            obs = batch["observation"].to(device).float()
            actions = batch["action"].to(device).float()
            pred = policy(obs)
            loss = F.mse_loss(pred, actions)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[bc] epoch {epoch:3d}/{args.epochs}  loss={total_loss / len(loader):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "obs_dim": obs_dim, "action_dim": action_dim}, out)
    print(f"[bc] saved {out}")


def _load_lerobot_dataset(repo_id: str, task: str):
    """Load a LeRobot dataset from HuggingFace Hub."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset(repo_id, task=task)
    except ImportError as exc:
        raise ImportError(
            "LeRobot not installed. Run: pip install lerobot\n"
            "See https://github.com/huggingface/lerobot"
        ) from exc


def _infer_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    obs = sample["observation"]
    action = sample["action"]
    import torch
    obs_dim = int(torch.as_tensor(obs).flatten().shape[0])
    action_dim = int(torch.as_tensor(action).flatten().shape[0])
    return obs_dim, action_dim


def main():
    args = _parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
