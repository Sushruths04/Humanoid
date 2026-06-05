# P2 — World Model (Dreamer-mini) on Isaac Nav Rollouts

Task: `Humanoid-G1-CommandNav-v0` rollouts → Dreamer-mini RSSM world model.
The world model learns dynamics from real policy rollouts and can roll forward
"in imagination" without running the simulator.

## Data Collection

- Policy: P0-stable checkpoint (`g1_commandnav_stable/model_499.pt`)
- 200 episodes collected (64 parallel envs), nav obs slice (4-dim: one-hot + rel_xy)
- Action dim: 37 (full G1 joint commands)
- Rollout file: `programs/data/nav_rollouts_commandnav.pt` (59 MB)

## World Model Training

| Metric | Value |
| --- | ---: |
| Architecture | RSSM: deter=128, stoch=32, hidden=128 |
| Training steps | 2000 |
| Batch size | 32 |
| Sequence length | 16 |
| Initial loss | 0.7625 |
| Final loss | **0.0109** (18× reduction) |
| Loss breakdown (final) | recon=0.0035, rew=0.0003, kl=0.0071 |

## Imagination Evaluation

| Metric | Value |
| --- | ---: |
| Imagined mean reward | **0.133** |
| Real mean reward | 0.144 |
| Imagined reward finite | **True** ✅ |

## Definition of Done

- Imagined reward is finite: **MET** ✅

## Reproduce

```bash
# Step 1: Collect rollouts (needs GPU + Isaac container)
bash programs/scripts/collect_nav_rollouts_cmd.sh   # or manually:
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/programs/world_model/collect_nav_rollouts.py \
  --task Humanoid-G1-CommandNav-v0 \
  --checkpoint /workspace/programs/checkpoints/g1_commandnav_stable/model_499.pt \
  --num-envs 64 --num-episodes 200 --obs-keys nav \
  --out /workspace/programs/data/nav_rollouts_commandnav.pt --headless

# Step 2: Train world model (CPU/GPU, no Isaac Sim needed)
python -u -m programs.world_model.train_wm_isaac \
  --data programs/data/nav_rollouts_commandnav.pt \
  --steps 2000 \
  --out programs/checkpoints/world_model/wm_commandnav.pt
```

Checkpoint: `mitvho09/humanoid-g1-nav` → `checkpoints/world_model/wm_commandnav.pt`
