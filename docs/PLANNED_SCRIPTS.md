# Planned Scripts — Inventory & GPU Test Runbook

Branch: `feat/planned-scripts`. This documents everything implemented while GPU
was unavailable, so it can be batch-tested in one pass when GPU returns.

## Status legend
- ✅ CPU-tested — pure logic, unit tests pass on CPU now.
- 🟡 GPU-pending — code complete + CPU import-checked; needs a GPU sim run.
- 🧩 Scaffold — structure + docs; needs an external env (graphics/Cosmos/manip).

## Inventory

| Area | File | What it does | Status |
|---|---|---|---|
| Steering | `programs/common/commands.py::velocity_command_to_target_avoiding` | Potential-field obstacle-avoiding steering | ✅ |
| Reward | `programs/common/rewards.py::collision_penalty` | Smooth proximity collision penalty | ✅ |
| Sequence | `programs/common/sequence.py` | `advance_subgoal`, `sample_subgoal_sequence` | ✅ |
| CP1.3 task | `tasks/g1_obstacle_nav_cfg.py` → `Humanoid-G1-ObstacleNav-v0` | Obstacle-aware nav (avoiding steer, obstacle obs, collision penalty) | 🟡 |
| CP1.4 task | `tasks/g1_seq_nav_cfg.py` → `Humanoid-G1-SeqNav-v0` | Sequential multi-goal (phase tracking, per-subgoal reward) | 🟡 |
| P2 world model | `programs/world_model/{rssm,agent,train_wm}.py` | Dreamer-mini RSSM + imagination actor-critic + toy trainer | ✅ |
| Launchers | `programs/scripts/{train_eval_nav,batch_test_nav,graphics_preflight}.sh` | Train+eval any nav task; batch all; Vulkan gate | 🟡 |
| P3 vision | `programs/vision/README.md` | Vision-conditioned nav plan + test cmd | 🧩 |
| P4 Cosmos | `programs/cosmos/{README,export_data,post_train}.py` | Cosmos post-train pipeline scaffold | 🧩 |
| T-track | `programs/manipulation/README.md` | Manipulation track plan | 🧩 |

## Run all CPU tests now
```bash
/home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest programs/ -q
# expect: all green (rewards, commands, sequence, text_encoder, world_model)
```

## GPU batch-test runbook (when GPU is back)

1. Bring up the container (image pull + this, see project memory):
```bash
cd ~/Humanoid/IsaacLab/docker && touch .isaac-lab-docker-history
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base -f docker-compose.yaml \
  --profile base up isaac-lab-base -d --no-build
```
2. Smoke-test all nav tasks (wiring check, ~minutes):
```bash
SMOKE=1 bash programs/scripts/batch_test_nav.sh
```
3. Full train + eval all nav tasks (each ~25 min on L4):
```bash
bash programs/scripts/batch_test_nav.sh         # writes docs/results/*.md
```
4. World model (no GPU needed, but verifies on the box):
```bash
python programs/world_model/train_wm.py --steps 300
```
5. Vision (only after graphics gate passes):
```bash
bash programs/scripts/graphics_preflight.sh
bash programs/scripts/train_eval_nav.sh Humanoid-G1-Vision-VLA-v0 128 300 16
```

## DoD targets (nav)
| Task | Success | Other |
|---|---|---|
| ObstacleNav | goal >= 65% | collision rate < 15% |
| SeqNav | two-step full-sequence >= 50% | ordering >= 70% |
| (ref) CommandNav | 94.5% achieved | — |
| (ref) LangNav | 98.8% per-command achieved | — |

## Notes / known design choices
- ObstacleNav obstacles are penalty regions (state-based); physical+visual
  obstacles arrive with the vision phase.
- All nav tasks reuse the verified P0 steering/reward/reset helpers; only the
  observation and (for obstacles/sequence) the per-step event differ.
