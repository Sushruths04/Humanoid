from huggingface_hub import HfApi
import sys

repo_id = "mitvho09/G1-Humanoid-VLA"
local_dir = "/home/zeus/content/Humanoid/thesis/checkpoints/g1_custom"

api = HfApi()

print(f"Uploading G1 production checkpoint to {repo_id}...")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model"
    )
    print("G1 Production Upload complete!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
