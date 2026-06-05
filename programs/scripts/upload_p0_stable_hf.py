"""Upload the P0-stable checkpoint and result to HuggingFace.

Usage (from repo root, in the conda env that has huggingface_hub):
    python programs/scripts/upload_p0_stable_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path


HF_REPO = "mitvho09/humanoid-g1-nav"
CKPT_LOCAL = "programs/checkpoints/g1_commandnav_stable/model_499.pt"
RESULT_LOCAL = "docs/results/p0_stable.md"

HF_CKPT_PATH = "checkpoints/g1_commandnav_stable/model_499.pt"
HF_RESULT_PATH = "results/p0_stable.md"


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    root = Path(__file__).parent.parent.parent  # repo root

    ckpt = root / CKPT_LOCAL
    result = root / RESULT_LOCAL

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run collect_p0_stable.sh first.")
    if not result.exists():
        raise FileNotFoundError(f"Result doc not found: {result}. Run collect_p0_stable.sh + evaluate.py first.")

    print(f"[hf] uploading checkpoint ({ckpt.stat().st_size // 1024 // 1024} MB) → {HF_CKPT_PATH}")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=HF_CKPT_PATH,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="P0-stable checkpoint (upright_reward=0.5, 500 iters)",
    )

    print(f"[hf] uploading result doc → {HF_RESULT_PATH}")
    api.upload_file(
        path_or_fileobj=str(result),
        path_in_repo=HF_RESULT_PATH,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="P0-stable eval results",
    )

    print(f"[hf] done. View at https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
