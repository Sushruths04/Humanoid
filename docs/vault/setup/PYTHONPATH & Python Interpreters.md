---
tags: [setup, python, pythonpath, gotcha]
---

# PYTHONPATH & Python Interpreters

## The Two Pythons — Never Mix Them

There are two completely separate Python environments. Using the wrong one is silent and causes confusing errors.

| Context | Command | Use for |
|---|---|---|
| **Isaac Sim (in container)** | `/workspace/isaaclab/isaaclab.sh -p <script.py>` | Anything that imports `isaaclab`, trains, evals |
| **CPU-only / unit tests (host)** | `/home/zeus/miniconda3/envs/cloudspace/bin/python` | Running `pytest`, CPU logic in `programs/`, py_compile checks |

> **Bare `python3` hits the wrong system Python** — it looks like it works but can't import `isaaclab` or your conda packages. Never use it.

---

## PYTHONPATH for In-Container Runs

Every `docker exec` call that runs a Python script must set this:

```bash
PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
```

What each path adds:
- `/workspace` — makes `programs.common.*` importable
- `/workspace/my-humanoid-project` — makes `my_humanoid_project.tasks` importable  
- `/workspace/isaaclab/source` — Isaac Lab source packages

**Template for any in-container run:**
```bash
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/<your_script.py> \
  <args>
```

---

## CPU Unit Tests (host, no container needed)

```bash
# Always use the full conda python path
cd /teamspace/studios/this_studio/Humanoid/programs
/home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest -q

# Syntax-check a file before scp-ing:
/home/zeus/miniconda3/envs/cloudspace/bin/python -m py_compile programs/common/eval/metrics.py && echo OK
```

Current baseline: **34 tests pass** (rewards, commands, sequence, text encoder, metrics including `sequence_eval_metrics`, world model).

---

## Task Registration Trick

Custom tasks need `import my_humanoid_project.tasks` to run before any Isaac Sim code. Both `custom_train.py` and `custom_play.py` do this at the top before importing the Isaac Lab trainer/player scripts. If you forget, you get `gym.error.UnregisteredEnv`.

---

## Related

- [[Isaac Sim Docker Container]]
- [[Lightning Studio Environment]]
- [[Training Recipe]]
