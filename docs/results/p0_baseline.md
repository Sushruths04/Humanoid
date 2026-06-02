# P0 Command-Nav Baseline — Results

Task: `Humanoid-G1-CommandNav-v0` (Unitree G1, Isaac Lab, RSL-RL PPO).
Genuine command-conditioned navigation: per episode a target marker is sampled
and markers are randomized; the policy is rewarded for reaching the COMMANDED
marker, and the base-velocity command is steered toward it.

## Training (4096 envs, 500 iters, NVIDIA L4)

| Metric | Early (iter ~5) | Final (iter 499) |
| --- | ---: | ---: |
| Mean episode length | 47 | 939 / ~1000 |
| `nav_command` reward | -0.0001 | +7.35 |
| `track_lin_vel_xy_exp` | 0.007 | 0.27 |
| Mean total reward | -6.3 | +91.9 |

## Evaluation (256 episodes)

| Metric | Value |
| --- | ---: |
| Commanded-target success rate | **0.945** |
| Success by command [red, blue] | [0.958, 0.934] |
| Fall rate | 0.281 |
| Mean final distance (m) | 0.348 |
| Mean episode length | 768 |

## Definition of Done

- Commanded-target success >= 0.70: **MET (0.945)**.
- Fall rate < 0.10: not yet (0.281) — robot often topples after reaching the
  target. Candidate fixes: more iterations, stronger stability reward weights,
  stop/settle bonus near target. Tracked as a follow-up; does not block P0.

## Reproduce

```bash
# train
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-CommandNav-v0 --headless --num_envs 4096 --max_iterations 500
# evaluate
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/programs/common/eval/evaluate.py \
  --task Humanoid-G1-CommandNav-v0 --headless --num-envs 256 --checkpoint <run>/model_499.pt
```

Checkpoint: `thesis/checkpoints/g1_commandnav/model_499.pt` (mirror to Hugging Face).
