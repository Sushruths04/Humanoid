"""SCAFFOLD (P4): export G1 (frames, actions) rollouts to the Cosmos cookbook
action-conditioned dataset format. Requires the Cosmos environment to finalize.

Intended CLI:
    python programs/cosmos/export_data.py --task Humanoid-G1-Vision-VLA-v0 \
        --checkpoint <ckpt> --out datasets/cosmos_g1 --episodes 200
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    raise NotImplementedError(
        "Cosmos data export scaffold. Roll out the policy, collect (frame_t, "
        "action_t, frame_t+1), and write the cookbook RoboCasa/Libero format to "
        f"{args.out}. Finalize against nvidia-cosmos/cosmos-predict2.5."
    )


if __name__ == "__main__":
    main()
