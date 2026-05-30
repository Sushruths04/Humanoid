# STEP 05 blocked

_2026-05-30T20:14:25Z_

No GR00T checkpoint found under thesis/checkpoints/gr00t.

# STEP 05 GR00T eval smoke

_2026-05-30T20:48:42Z_

# STEP 05 GR00T eval smoke

_2026-05-30T20:48:42Z_

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python gr00t/eval/open_loop_eval.py --dataset-path /home/zeus/content/Humanoid/Isaac-GR00T/demo_data/cube_to_bowl_5 --embodiment-tag NEW_EMBODIMENT --model-path /home/zeus/content/Humanoid/thesis/checkpoints/gr00t_smoke/checkpoint-2000 --traj-ids 0 --steps 64 --action-horizon 16 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_eval_smoke/traj_0.jpeg --modality-keys single_arm gripper
```

# STEP 05 result

_2026-05-30T20:49:05Z_


## Result

```text
# GR00T eval smoke summary

Generated: 2026-05-30T20:49:05Z
Exit status: 0
Checkpoint: /home/zeus/content/Humanoid/thesis/checkpoints/gr00t_smoke/checkpoint-2000
Dataset: /home/zeus/content/Humanoid/Isaac-GR00T/demo_data/cube_to_bowl_5
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_eval_smoke/traj_0.jpeg
Log: /home/zeus/content/Humanoid/thesis/logs/05_gr00t_eval.log

## Key lines
INFO:root:Dataset length: 5
INFO:root:Using 64 steps (requested: 64, trajectory length: 568)
INFO:root:Unnormalized Action MSE across single traj: 25.878576278686523
INFO:root:Unnormalized Action MAE across single traj: 3.016110420227051
INFO:root:MSE for trajectory 0: 25.878576278686523, MAE: 3.016110420227051
INFO:root:Average MSE across all trajs: 25.878576278686523
INFO:root:Average MAE across all trajs: 3.016110420227051
INFO:root:Done
```

