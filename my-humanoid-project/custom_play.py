"""Entry point for PLAYING/RECORDING custom humanoid tasks in Isaac Lab.

Mirrors custom_train.py: register our gym envs first, then hand off to the
stock RSL-RL play.py (which launches the sim app + records video when --video
is passed). Video is written to <log_dir>/videos/play/.
"""

import os
import sys

# Register our tasks first (string entry points -> safe without Isaac Sim).
try:
    import my_humanoid_project.tasks  # noqa: F401
    print("Successfully registered my_humanoid_project.tasks")
except ImportError as e:
    print(f"Error importing my_humanoid_project.tasks: {e}")
    sys.exit(1)

# Make the stock RSL-RL play.py importable.
isaaclab_path = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")
rsl_rl_path = os.path.join(isaaclab_path, "scripts", "reinforcement_learning", "rsl_rl")
if rsl_rl_path not in sys.path:
    sys.path.insert(0, rsl_rl_path)

# Importing play launches the Omniverse app at module level (parses sys.argv).
try:
    from play import main, simulation_app
    print("Successfully imported Isaac Lab RSL-RL play")
except ImportError as e:
    print(f"Error importing Isaac Lab play: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
    simulation_app.close()
