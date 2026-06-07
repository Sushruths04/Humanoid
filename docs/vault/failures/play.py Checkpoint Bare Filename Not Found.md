---
tags: [failure, eval, play, checkpoint, isaac-lab, p3]
---

# play.py `--checkpoint` Bare Filename Not Found

## Symptom
Running eval with `--checkpoint model_499.pt` fails:
```
FileNotFoundError: Unable to find the file: model_499.pt
```
The file physically exists in the logs directory, but Isaac Lab can't find it.

## Root Cause
`play.py` has two code paths for resolving the checkpoint:
```python
elif args_cli.checkpoint:
    resume_path = retrieve_file_path(args_cli.checkpoint)   # path lookup — bare name fails
else:
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

When `--checkpoint model_499.pt` is passed, `retrieve_file_path("model_499.pt")` is called. This function checks if the string is a valid file path — a bare filename without directory is not a valid path, so it raises `FileNotFoundError`.

The `--load_run` flag correctly routes to `get_checkpoint_path`, which searches `logs/rsl_rl/<experiment>/<load_run>/` for matching files.

## Fix
**Option A (recommended):** Use only `--load_run`, omit `--checkpoint`. Isaac Lab finds the latest `.pt` file in the run directory automatically:
```bash
docker exec ... custom_play.py \
  --task Humanoid-G1-VisionNav-v0 \
  --load_run p3_eval    # get_checkpoint_path finds model_499.pt automatically
```

**Option B:** Pass the full absolute path to `--checkpoint`:
```bash
--checkpoint /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/model_499.pt
```

## Secondary Failure: OOM During Eval on L4
Even after fixing the path, eval with `--num_envs 512` OOM'd on L4-24G because the training process (PID 34) was still alive in the container holding ~20 GB VRAM even after training finished. Fix:
```bash
docker exec container_name bash -c 'pkill -9 -f custom_train; pkill -9 -f isaaclab'
# Verify VRAM freed:
docker exec container_name nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```
Then rerun eval with fewer envs (`--num_envs 128`) as a safety margin.

## Rule
> **Never use `--checkpoint <filename>` alone.** Either use `--load_run` only (cleanest) or pass a full absolute path. Always kill the training process before running eval on the same machine.

## Related
- [[play.py Fails - Custom Task Not Registered]]
- [[RSL-RL Resume Resets Loop Counter]]
- [[OOM With Camera Rollout Buffer]]
