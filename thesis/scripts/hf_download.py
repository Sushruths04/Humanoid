from huggingface_hub import snapshot_download
import sys

repo_id = "mitvho09/GR00T-Humanoid"
local_dir = "/home/zeus/content/Humanoid/thesis/checkpoints/gr00t_smoke"

print(f"Downloading {repo_id} to {local_dir}...")
try:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("Download complete!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
