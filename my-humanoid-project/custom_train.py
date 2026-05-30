"""Entry point for training custom humanoid tasks in Isaac Lab."""

import os
import sys

# Import our tasks first to register them in Gymnasium
# Registration now uses string entry points, so it's safe to import without Isaac Sim
try:
    import my_humanoid_project.tasks
    print("Successfully registered my_humanoid_project.tasks")
except ImportError as e:
    print(f"Error importing my_humanoid_project.tasks: {e}")
    sys.exit(1)

# Import the Isaac Lab RSL-RL training script
isaaclab_path = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")
rsl_rl_path = os.path.join(isaaclab_path, "scripts", "reinforcement_learning", "rsl_rl")
if rsl_rl_path not in sys.path:
    sys.path.insert(0, rsl_rl_path)

try:
    from train import main
    print("Successfully imported Isaac Lab RSL-RL trainer")
except ImportError as e:
    print(f"Error importing Isaac Lab trainer: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
