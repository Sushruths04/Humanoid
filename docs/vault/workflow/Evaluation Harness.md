---
tags: [workflow, evaluation, metrics, harness]
---

# Evaluation Harness

## Overview

All evaluation is in `programs/common/eval/`. It's simulator-agnostic pure Python — fully CPU-unit-testable.

```
programs/common/eval/
  metrics.py        ← pure aggregate functions (CPU-tested)
  report.py         ← write_results_markdown()
  evaluate.py       ← single-target rollout (CommandNav, LangNav, ObstacleNav)
  evaluate_seq.py   ← sequential rollout (SeqNav)
```

---

## metrics.py

### For single-target tasks

```python
from programs.common.eval.metrics import compute_episode_metrics, success_rate_by_command

metrics = compute_episode_metrics(reached, fell, final_distance, episode_length)
# → {'num_episodes': 256, 'success_rate': 0.859, 'fall_rate': 0.242,
#    'mean_final_distance': 0.63, 'mean_episode_length': 827}

by_cmd = success_rate_by_command(reached, command_ids, num_commands=3)
# → tensor([0.837, 0.880])  — per-command success rates
```

### For sequential tasks

```python
from programs.common.eval.metrics import sequence_eval_metrics

# reach_steps[i, k] = first step subgoal k was reached (-1 if never)
metrics = sequence_eval_metrics(reach_steps, num_subgoals=2)
# → {'full_sequence_success': 0.809, 'ordering_accuracy': 0.945,
#    'first_subgoal_rate': 0.977, ...}
```

**Full sequence = ALL subgoals reached AND in the commanded order** (non-decreasing reach steps).  
**Ordering = of episodes that reached all subgoals, what fraction were in order.**

---

## Running evaluate.py (single-target tasks)

```bash
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
docker exec -e PYTHONPATH="$PP" isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/programs/common/eval/evaluate.py \
  --task Humanoid-G1-ObstacleNav-v0 \
  --headless \
  --num-envs 256 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/model_499.pt \
  --out programs/results/humanoid-g1-obstaclenav-v0.md
```

> **Note:** writes to `programs/results/` (bind-mounted → host). The `train_eval_nav.sh` script does this automatically.

---

## Running evaluate_seq.py (SeqNav only)

```bash
docker exec -e PYTHONPATH="$PP" isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/programs/common/eval/evaluate_seq.py \
  --task Humanoid-G1-SeqNav-v0 \
  --headless \
  --num-envs 256 \
  --checkpoint /workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/model_499.pt \
  --out programs/results/humanoid-g1-seqnav-v0.md
```

> **Do NOT use `evaluate.py` for SeqNav** — it either crashes (`AttributeError`) or measures wrong metrics. See [[Eval Crash - Missing Buffer]].

---

## The `handle_deprecated_rsl_rl_cfg` Fix

Both evaluators include this critical line before loading the checkpoint:

```python
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
import importlib.metadata as _metadata
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
```

Without it: `KeyError: 'class_name'` when RSL-RL 2.x tries to deserialize the checkpoint config.

---

## CPU Unit Tests

```bash
cd programs
/home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest common/tests/test_metrics.py -v
# Tests: compute_episode_metrics, success_rate_by_command, sequence_eval_metrics
# All are pure tensor functions — no Isaac Sim needed
```

---

## Related

- [[Training Recipe]]
- [[PPO with RSL-RL]]
- [[Eval Crash - Missing Buffer]]
- [programs/common/eval/metrics.py](../../programs/common/eval/metrics.py)
- [programs/common/eval/evaluate_seq.py](../../programs/common/eval/evaluate_seq.py)
