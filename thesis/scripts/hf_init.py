from huggingface_hub import HfApi
import sys

repo_id = "mitvho09/G1-Humanoid-VLA"

api = HfApi()

print(f"Ensuring repository {repo_id} exists...")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print("Repository is ready.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
