---
tags: [failure, eval, play, gymnasium, task-registration, p3]
---

# play.py Fails — Custom Task Not Registered

## Symptom
Ran Isaac Lab's stock `play.py` to evaluate `model_499.pt`:
```bash
docker exec ... /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Humanoid-G1-VisionNav-v0 ...
```
Got:
```
gymnasium.error.NameNotFound: Environment `Humanoid-G1-VisionNav` doesn't exist.
```
Even with `PYTHONPATH` set correctly.

## Root Cause
Isaac Lab's `play.py` calls `gym.spec(task_name)` to load the environment config. This requires the task to be **registered** with Gymnasium via `gym.register(...)`. 

Our custom task `Humanoid-G1-VisionNav-v0` is registered when `my_humanoid_project.tasks` is imported. But `play.py` never imports this module — only `custom_train.py` does it explicitly.

Setting `PYTHONPATH` makes the package importable but doesn't trigger the import. Python only registers the task when the code runs `import my_humanoid_project.tasks`.

## Fix: custom_play.py
Mirror the pattern from `custom_train.py` — register tasks first, then delegate:

```python
# my-humanoid-project/custom_play.py
import os, sys

try:
    import my_humanoid_project.tasks  # THIS registers Humanoid-G1-VisionNav-v0
    print("Successfully registered my_humanoid_project.tasks")
except ImportError as e:
    print(f"Error: {e}"); sys.exit(1)

isaaclab_path = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")
rsl_rl_path = os.path.join(isaaclab_path, "scripts", "reinforcement_learning", "rsl_rl")
if rsl_rl_path not in sys.path:
    sys.path.insert(0, rsl_rl_path)

from play import main, simulation_app
if __name__ == "__main__":
    main()
    simulation_app.close()
```

Use it exactly like `custom_train.py`:
```bash
docker exec ... /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 512 \
  --load_run <timestamp_dir> \
  --checkpoint model_499.pt
```

## Rule
> For ANY custom Isaac Lab task, use `custom_train.py` / `custom_play.py` as the entry point. Never call `train.py` or `play.py` directly.

Both scripts live at `my-humanoid-project/` and are the canonical entry points.

## Related
- [[SSH Heredoc Apostrophe Corruption]] — second failure when writing custom_play.py over SSH
- [[Python File Corruption Over SSH - Use Python Write]]
- [[Training Recipe]]
- [[Evaluation Harness]]
