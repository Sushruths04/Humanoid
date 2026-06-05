# T0 — Manipulation Foundation

**Status:** metrics + eval harness built (CPU-verified). Env integration pending.

## What's here

- `evaluate_manip.py` — eval harness: rolls out a manipulation policy, records
  grasp/place/drop/task-success per episode, writes `docs/results/t0_manip.md`
- `programs/common/eval/manip_metrics.py` — pure tensor metrics (10 tests, all green)

## Setup (one-time, on the GPU machine)

### 1. Install LIBERO inside the container

```bash
docker exec -it isaac-lab-base bash
pip install libero-benchmark  # or from source: git clone + pip install -e .
```

### 2. Verify LIBERO tasks load

```python
from libero.libero import benchmark
b = benchmark.get_benchmark_dict()
print(list(b.keys()))  # should list libero_spatial, libero_object, etc.
```

### 3. Run a scripted baseline (sanity check)

```bash
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/programs/t0_manip_foundation/evaluate_manip.py \
  --task libero_spatial \
  --num-envs 64 \
  --out programs/results/t0_manip.md
```

## Definition of Done (CPT0.2)

`evaluate.py --suite libero --ckpt <x>` prints metric dict + writes `docs/results/t0_manip.md`

Pass gate: task_success > 0 (env runs, not all crashes) AND harness writes markdown.

## Metrics captured

| Metric | Description |
|---|---|
| grasp_success | Robot achieved stable grasp (object lifted > threshold) |
| place_success | Object placed within place_radius of target |
| task_success | grasped AND placed AND not dropped |
| object_drop_rate | Robot dropped object after grasping |
| mean_steps_to_success | Steps until task_success (over successful episodes only) |
