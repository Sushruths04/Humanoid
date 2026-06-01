#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Humanoid artifacts from Hugging Face.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "<your-hf-namespace>/Humanoid-VLA-Artifacts"))
    parser.add_argument("--path-in-repo", default=os.environ.get("HF_PATH_IN_REPO", "phase3"))
    parser.add_argument("--local-dir", default=os.environ.get("HF_LOCAL_DIR", "thesis/artifacts/phase3_download"))
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required to download artifacts.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    os.makedirs(args.local_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision="main",
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"{args.path_in_repo}/**"],
    )
    print(f"Downloaded {args.repo_id}/{args.path_in_repo} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
