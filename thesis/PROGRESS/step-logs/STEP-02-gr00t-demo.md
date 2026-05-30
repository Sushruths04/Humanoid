# STEP 02 GR00T demo command gate

_2026-05-30T18:58:36Z_

## Ready to run

GPU detected. Next action is to verify the current Isaac-GR00T README and run its official inference demo.

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
nvidia-smi
# Run the current repo's documented GR00T-N1-2B inference command here.
```

Do not mark this step done until an action tensor/output shape is captured.

# STEP 02 GR00T GPU demo

_2026-05-30T19:02:02Z_

# STEP 02 GR00T GPU demo

__

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:02:31Z_


## Result

```text
# GR00T demo summary

Generated: 
Exit status: 1
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Key lines
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:03:54Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:03:54Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:04:04Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:04:04Z
Exit status: 1
Blocked by gated Hugging Face access: 1
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present False

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Access to model nvidia/Cosmos-Reason2-2B is restricted. You must have access to it and be authenticated to access it. Please log in.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Access to model nvidia/Cosmos-Reason2-2B is restricted. You must have access to it and be authenticated to access it. Please log in.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present False
```

# STEP 02 waiting on HF auth

_2026-05-30T19:04:04Z_

## Waiting for Hugging Face authentication

The GPU, Torch CUDA, GR00T import, and policy import passed. Model loading is blocked because the checkpoint depends on gated repo nvidia/Cosmos-Reason2-2B.

```bash
hf auth login
# then rerun: bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:18:41Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:18:41Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:18:51Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:18:51Z
Exit status: 1
Blocked by gated Hugging Face access: 1
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present False

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Access to model nvidia/Cosmos-Reason2-2B is restricted. You must have access to it and be authenticated to access it. Please log in.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Access to model nvidia/Cosmos-Reason2-2B is restricted. You must have access to it and be authenticated to access it. Please log in.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present False
```

# STEP 02 waiting on HF auth

_2026-05-30T19:18:51Z_

## Waiting for Hugging Face authentication

The GPU, Torch CUDA, GR00T import, and policy import passed. Model loading is blocked because the checkpoint depends on gated repo nvidia/Cosmos-Reason2-2B.

```bash
hf auth login
# then rerun: bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:19:39Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:19:39Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:19:49Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:19:49Z
Exit status: 1
Blocked by gated Hugging Face access: 1
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present True

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present True
```

# STEP 02 waiting on HF auth

_2026-05-30T19:19:49Z_

## Waiting for Hugging Face authentication

The GPU, Torch CUDA, GR00T import, and policy import passed. Model loading is blocked because the checkpoint depends on gated repo nvidia/Cosmos-Reason2-2B.

```bash
hf auth login
# then rerun: bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:28:01Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:28:01Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:28:38Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:28:38Z
Exit status: 1
Blocked by gated Hugging Face access: 0
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present True

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Model loading time: 26.3830 seconds
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Dataset length: 3
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present True
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:31:11Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:31:11Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:31:39Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:31:39Z
Exit status: 0
Blocked by gated Hugging Face access: 0
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present True

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Model loading time: 13.3415 seconds
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Dataset length: 3
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Using 16 steps (requested: 16, trajectory length: 266)
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:pred_action_joints vs time (16, 17)
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Average MSE across all trajs: 0.000687
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Average MAE across all trajs: 0.014082
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Model loading time:          13.3415s
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Avg inference time per step: 0.2503s
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Done
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present True
```

# STEP 02 GR00T GPU demo

_2026-05-30T19:33:20Z_

# STEP 02 GR00T GPU demo

_2026-05-30T19:33:20Z_

## GPU

```text
NVIDIA L4, 23034 MiB, 22564 MiB
```

## Command

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
/home/zeus/miniconda3/envs/thesis310/bin/python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg
```

# STEP 02 result

_2026-05-30T19:33:51Z_


## Result

```text
# GR00T demo summary

Generated: 2026-05-30T19:33:51Z
Exit status: 0
Blocked by gated Hugging Face access: 0
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
torchcodec_import 0.4.0
video_decoder VideoDecoder
hf_token_present True

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Model loading time: 13.2580 seconds
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Dataset length: 3
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Using 16 steps (requested: 16, trajectory length: 266)
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:pred_action_joints vs time (16, 17)
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Average MSE across all trajs: 0.000687
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Average MAE across all trajs: 0.014082
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Model loading time:          13.2580s
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:  Avg inference time per step: 0.2409s
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:INFO:root:Done
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present True
```

