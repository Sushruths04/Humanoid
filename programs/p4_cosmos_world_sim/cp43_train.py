"""CP4.3 LoRA post-training of Cosmos Predict2 on G1 nav Bridge data.

Uses the real cosmos-predict2 training system via torchrun.
Run from /tmp/cosmos-predict2/ (megatron/imaginaire need that root).

Usage (smoke test, 2 steps):
    cd /tmp/cosmos-predict2
    torchrun --nproc_per_node=1 -m scripts.train         --config=cosmos_predict2/configs/base/config.py --         experiment="predict2_video2world_2b_action_conditioned_training"         trainer.max_iter=2         trainer.checkpoint_dir=/tmp/cosmos_smoke         dataset.data_path=/teamspace/studios/this_studio/Humanoid/datasets/g1_nav         model.lora_rank=16

Usage (full 5000 steps):
    cd /tmp/cosmos-predict2
    torchrun --nproc_per_node=1 -m scripts.train         --config=cosmos_predict2/configs/base/config.py --         experiment="predict2_video2world_2b_action_conditioned_training"         trainer.max_iter=5000         trainer.checkpoint_dir=/teamspace/studios/this_studio/Humanoid/checkpoints/p4_cosmos_lora         trainer.checkpoint_every=500         dataset.data_path=/teamspace/studios/this_studio/Humanoid/datasets/g1_nav         model.lora_rank=16

This script is a Python wrapper that builds & runs the torchrun command.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path


COSMOS_ROOT = "/tmp/cosmos-predict2"
BASE_CONFIG = "cosmos_predict2/configs/base/config.py"
EXPERIMENT = "predict2_video2world_2b_action_conditioned_training"
REPO_DIR = "/teamspace/studios/this_studio/Humanoid"
DIT_PATH = f"{REPO_DIR}/checkpoints/cosmos_base/model-480p-4fps.pt"


def build_torchrun_cmd(
    data_path: str,
    checkpoint_dir: str,
    max_iter: int = 5000,
    checkpoint_every: int = 500,
    lora_rank: int = 16,
    num_gpus: int = 1,
) -> list[str]:
    return [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "scripts.train",
        f"--config={BASE_CONFIG}",
        "--",
        f"experiment={EXPERIMENT}",
        f"trainer.max_iter={max_iter}",
        f"trainer.checkpoint_dir={checkpoint_dir}",
        f"trainer.checkpoint_every={checkpoint_every}",
        f"dataset.data_path={data_path}",
        f"model.lora_rank={lora_rank}",
        f"model.dit_path={DIT_PATH}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=f"{REPO_DIR}/datasets/g1_nav")
    parser.add_argument("--out", default=f"{REPO_DIR}/checkpoints/p4_cosmos_lora")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 2 iterations only")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(COSMOS_ROOT):
        print(f"[FAIL] cosmos-predict2 not found at {COSMOS_ROOT}")
        print("Run setup_machine.sh first.")
        sys.exit(1)

    max_iter = 2 if args.smoke else args.max_steps
    ckpt_dir = "/tmp/cosmos_smoke" if args.smoke else args.out

    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = build_torchrun_cmd(
        data_path=args.data,
        checkpoint_dir=ckpt_dir,
        max_iter=max_iter,
        checkpoint_every=2 if args.smoke else args.save_every,
        lora_rank=args.lora_rank,
        num_gpus=args.num_gpus,
    )

    print("Training command:")
    print(" ".join(cmd))
    print()

    # Run from cosmos root so megatron/imaginaire imports resolve
    result = subprocess.run(cmd, cwd=COSMOS_ROOT)

    if result.returncode != 0:
        print(f"[FAIL] Training exited with code {result.returncode}")
        sys.exit(result.returncode)

    if args.smoke:
        print("SMOKE TEST PASSED")
    else:
        print(f"CP4.3 DONE — LoRA checkpoint at {ckpt_dir}")


if __name__ == "__main__":
    main()
