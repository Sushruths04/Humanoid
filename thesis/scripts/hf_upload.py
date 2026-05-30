import os
from huggingface_hub import HfApi

# Configuration
REPO_ID = "mitvho09/GR00T-Humanoid"
LOCAL_DIR = "/home/zeus/content/Humanoid/thesis/checkpoints/gr00t_smoke"

api = HfApi()

print(f"Uploading {LOCAL_DIR} to {REPO_ID} (excluding older checkpoints)...")
try:
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=["checkpoint-[0-9]*", "!checkpoint-10000"]
    )
    print("Full upload complete!")
except Exception as e:
    print(f"Error during upload: {e}")
    exit(1)
