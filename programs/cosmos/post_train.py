"""SCAFFOLD (P4): LoRA post-train of Cosmos-Predict 2.5 on exported G1 data.

Smoke-gate hard before committing GPU (this is the budget-critical run). Use
LoRA + bf16 + gradient checkpointing + 8-bit optimizer. Requires the Cosmos env.

Intended CLI:
    python programs/cosmos/post_train.py --data datasets/cosmos_g1 \
        --base cosmos-predict2.5-2b --lora --bf16 --max-steps 2000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", default="cosmos-predict2.5-2b")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    raise NotImplementedError(
        "Cosmos LoRA post-train scaffold. Wire to the cosmos-cookbook Robot/Policy "
        "recipe. DoD: action-conditioned generation responds to different actions."
    )


if __name__ == "__main__":
    main()
