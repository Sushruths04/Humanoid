# T0 Manipulation Eval: libero_spatial:0

Task: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`  
Policy: `bc_libero_spatial_0.pt` (MLPBCPolicy, 200 epochs on 50 demos)  
Episodes: 20  

## Results

| Metric | Value |
|---|---|
| task_success | **0.500** |
| grasp_success | 0.700 |
| place_success | 0.500 |
| object_drop_rate | 0.700 |
| mean_steps_to_success | 109.2 |

## Definition of Done

- task_success > 0%: **MET** ✅ (50.0%)

## Architecture

| Parameter | Value |
|---|---|
| Policy | MLPBCPolicy (2-layer MLP) |
| obs_dim | 12 (joint_pos 7 + eef_pos 3 + gripper 2) |
| action_dim | 7 (OSC_POSE delta) |
| hidden | 256 |
| Training data | 50 demos × ~100 steps = 5018 transitions |
| Epochs | 200 |
| Final BC loss | 0.044 |

## Reproduce

```bash
# Step 1: Install LIBERO (conda env on Lightning Studio)
# See docs/vault/tasks/T0 - ManipFoundation.md for install details

# Step 2: Download demos
python /tmp/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir /teamspace/studios/this_studio/libero_datasets \
  --datasets libero_spatial --use-huggingface

# Step 3: Train BC policy
python -m programs.t0_manip_foundation.train_bc_libero \
  --data-dir /teamspace/studios/this_studio/libero_datasets/libero_spatial \
  --task-idx 0 --epochs 200 \
  --out programs/checkpoints/t0_bc/bc_libero_spatial_0.pt

# Step 4: Evaluate
MUJOCO_GL=egl python -m programs.t0_manip_foundation.evaluate_manip \
  --task libero_spatial:0 \
  --checkpoint programs/checkpoints/t0_bc/bc_libero_spatial_0.pt \
  --num-envs 20 --out docs/results/t0_manip.md
```
