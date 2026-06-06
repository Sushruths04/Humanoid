# GR00T N1.5 / N1.7 — Complete Robot Setup & Fine-Tuning Guide

A hands-on, interview-ready reference for deploying NVIDIA GR00T on any robot:
UR5, Franka Panda, SO-100, or a custom URDF robot.

---

## Table of Contents

1. [What GR00T Is (and How It Works)](#1-what-groot-is-and-how-it-works)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Understanding the Embodiment System](#4-understanding-the-embodiment-system)
5. [Defining Your Robot's Embodiment](#5-defining-your-robots-embodiment)
6. [Data Collection — Teleoperation](#6-data-collection--teleoperation)
7. [Dataset Formatting (LeRobot Format)](#7-dataset-formatting-lerobot-format)
8. [LoRA Fine-Tuning](#8-lora-fine-tuning)
9. [Evaluation](#9-evaluation)
10. [Real Robot Deployment](#10-real-robot-deployment)
11. [Using Different Base Models](#11-using-different-base-models)
12. [Common Pitfalls & Debugging](#12-common-pitfalls--debugging)
13. [Interview Quick-Reference](#13-interview-quick-reference)

---

## 1. What GR00T Is (and How It Works)

### Architecture

GR00T is a **Vision-Language-Action (VLA)** model — it takes images, robot state, and a natural-language task description as input, and outputs robot actions.

```
Input:
  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐
  │ RGB images  │   │ robot state │   │ "pick up the cup and │
  │ (cameras)   │   │ (eef, joints│   │  place it on plate"  │
  └──────┬──────┘   └──────┬──────┘   └──────────┬───────────┘
         │                 │                      │
         └─────────────────▼──────────────────────┘
                     ┌─────────────┐
                     │  Qwen3-VL   │  ← Vision-Language backbone (like GPT-4V)
                     │  (2.7B)     │     Fuses vision + language tokens
                     └──────┬──────┘
                            │ latent features
                     ┌──────▼──────┐
                     │  DiT Action │  ← Diffusion Transformer head
                     │  Head (0.3B)│     Denoises Gaussian noise → action
                     └──────┬──────┘
                            │
Output: action chunk (8 steps) = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

### Training Pipeline

| Stage | Who does it | Scale | Time |
|---|---|---|---|
| **Pre-training** | NVIDIA | Open-X Embodiment: 1M+ demos, 22 robot types | Months, 1000s of GPUs |
| **Robot fine-tune** | NVIDIA / you | Your robot demos (50-2000) | 4-24 hrs, 1× A100 |
| **Inference** | You | No training | Real-time, 1× GPU ≥16GB |

### Why It Works So Well

The model has already learned:
- What objects look like from images
- How grasping works physically
- Language grounding ("pick up", "place on", "open drawer")
- General manipulation priors from 22+ robot morphologies

Fine-tuning just adapts it to **your specific robot's kinematics and your specific camera setup**. Think of it like fine-tuning GPT-4 for a domain — the hard part is done.

---

## 2. Hardware Requirements

| Operation | Min VRAM | Recommended | Notes |
|---|---|---|---|
| Inference only | 15 GB | L4 (24 GB) | T4 works with `--denoising-steps 4` |
| LoRA fine-tuning | 40 GB | L40S (48 GB) | Single GPU |
| Full fine-tuning | 80 GB | A100-80G | 8× for paper-scale |
| Data collection | CPU only | Any laptop | Teleoperation on local machine |

**Cloud rentals (2026 prices):**
- L4 24GB — ~$0.50/hr (Lambda, RunPod, Lightning AI)
- L40S 48GB — ~$1.50/hr
- A100-80G — ~$3/hr

---

## 3. Environment Setup

### Step 1 — Clone Isaac-GR00T

```bash
# Clone the official repo (always use latest main)
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
```

### Step 2 — Create Conda Environment

```bash
# Python 3.10 required (3.11+ not tested)
conda create -n groot_env python=3.10 -y
conda activate groot_env

# Install PyTorch with CUDA 12.x
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Install GR00T and all dependencies
pip install -e '.' --no-build-isolation

# Install flash-attention (required for Qwen3-VL, must use prebuilt wheel)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fix numpy version conflict (gym pulls numpy 2.x, gr00t needs 1.26.4)
pip install 'numpy==1.26.4'
```

### Step 3 — Verify Installation

```python
import gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.model.transforms import VideoToTensor
print("GR00T install OK")
```

### Step 4 — Download Pre-Trained Checkpoint

```bash
# For LIBERO (Franka Panda in simulation)
huggingface-cli download nvidia/GR00T-N1.7-LIBERO \
    --local-dir ./checkpoints/groot_libero \
    --include "libero_spatial/**"

# For SO-100 arm (community checkpoint)
huggingface-cli download nvidia/GR00T-N1.5-3B \
    --local-dir ./checkpoints/groot_base

# General base model (fine-tune from this for new robots)
huggingface-cli download nvidia/GR00T-N1.7-3B \
    --local-dir ./checkpoints/groot_n17_base
```

---

## 4. Understanding the Embodiment System

GR00T uses an **embodiment tag** system to handle different robot morphologies. Each embodiment defines:

- What observations the robot provides (state keys + shapes)
- What actions it takes (action keys + shapes)
- How many cameras it has

### Key Files

```
Isaac-GR00T/
├── gr00t/data/embodiment_tags.py     ← EmbodimentTag enum (all supported robots)
├── examples/LIBERO/modality.json     ← obs/action schema for LIBERO Franka
├── examples/SO100/modality.json      ← obs/action schema for SO-100
└── gr00t/eval/sim/LIBERO/libero_env.py  ← reference: how obs→model works
```

### Built-in Embodiment Tags

```python
from gr00t.data.embodiment_tags import EmbodimentTag

# Access by NAME not value:
tag = EmbodimentTag["LIBERO_PANDA"]   # ✅ correct
tag = EmbodimentTag("LIBERO_PANDA")   # ❌ raises ValueError

# Common tags:
# LIBERO_PANDA  — Franka Panda in LIBERO sim
# GR1           — FFTAI GR1 humanoid
# G1            — Unitree G1 humanoid
# SO100         — SO-100 5-DOF desktop arm
# AGILEX_LEROBOT — AgileX mobile manipulator
```

### modality.json — The Schema File

This is the most important file. It defines the exact tensor shapes and keys:

```json
{
  "observation": {
    "video": {
      "image": {"original_shape": [3, 256, 256], "info": "main camera RGB"},
      "wrist_image": {"original_shape": [3, 256, 256], "info": "wrist camera RGB"}
    },
    "state": {
      "x": {"original_shape": [1], "info": "eef x position"},
      "y": {"original_shape": [1], "info": "eef y position"},
      "z": {"original_shape": [1], "info": "eef z position"},
      "roll": {"original_shape": [1], "info": "axis-angle x"},
      "pitch": {"original_shape": [1], "info": "axis-angle y"},
      "yaw": {"original_shape": [1], "info": "axis-angle z"},
      "gripper": {"original_shape": [2], "info": "gripper joint positions"}
    },
    "language": {
      "annotation.human.action.task_description": {}
    }
  },
  "action": {
    "x": {"original_shape": [1]},
    "y": {"original_shape": [1]},
    "z": {"original_shape": [1]},
    "roll": {"original_shape": [1]},
    "pitch": {"original_shape": [1]},
    "yaw": {"original_shape": [1]},
    "gripper": {"original_shape": [1]}
  }
}
```

**Critical conventions (get these wrong → 0% success):**
- State rotation: **axis-angle**, NOT Euler angles
- Quaternion input: convert `[w,x,y,z]` → `[x,y,z,w]` before converting to axis-angle
- Images: passed as `(B=1, T=1, H, W, 3)` uint8 tensors
- Images must be **flipped both axes** (`[::-1, ::-1]`) to match training distribution
- Gripper action: GR00T outputs `[0,1]` (0=close), LIBERO needs `[-1,+1]` inverted

---

## 5. Defining Your Robot's Embodiment

### For Franka Panda (pre-built)

Already supported as `LIBERO_PANDA`. Use the LIBERO checkpoint directly.

### For UR5

UR5 is **not** in the default embodiment list. You have two options:

**Option A — Map to closest existing embodiment**

UR5 has 6 DOF + gripper. Treat it like a Franka (7 DOF → ignore redundant joint):
- Use `LIBERO_PANDA` tag
- State: eef_xyz(3) + axis_angle(3) + gripper(2) = 8-dim (same schema)
- Collect demos with UR5, fine-tune — the model adapts to UR5 kinematics through the data

**Option B — Define custom embodiment**

```python
# gr00t/data/embodiment_tags.py — add to EmbodimentTag enum:
UR5_ROBOTIQ = "ur5_robotiq"

# Create examples/UR5/modality.json:
{
  "observation": {
    "video": {
      "image": {"original_shape": [3, 480, 640]},
      "wrist_image": {"original_shape": [3, 480, 640]}
    },
    "state": {
      "joint_pos": {"original_shape": [6]},   # UR5 joint angles
      "gripper": {"original_shape": [1]}       # Robotiq 85
    }
  },
  "action": {
    "joint_vel": {"original_shape": [6]},
    "gripper": {"original_shape": [1]}
  }
}
```

### For SO-100 (community supported)

```bash
# Already has examples in Isaac-GR00T:
ls Isaac-GR00T/examples/SO100/

# modality.json is there, use tag:
tag = EmbodimentTag["SO100"]
```

### For Custom URDF Robot

```
Step 1: Decide your control space
  - End-effector OSC (position + orientation + gripper) — easiest, model-agnostic
  - Joint space (velocity or position) — more robot-specific

Step 2: Create modality.json (see template above)
  - Match your actual sensor outputs
  - Match your actual controller inputs

Step 3: Add embodiment tag to EmbodimentTag enum

Step 4: Collect demos with your actual robot

Step 5: Fine-tune (same as below)
```

---

## 6. Data Collection — Teleoperation

### Hardware Options

| Device | Cost | Best for |
|---|---|---|
| SpaceMouse (3Dconnexion) | ~$150 | Smooth 6-DOF control, professional |
| VR controllers (Quest 2) | ~$300 | Most natural, highest quality demos |
| Keyboard | Free | Crude, use only for testing |
| Gello (open-source leader arm) | ~$500 DIY | Best for real manipulation tasks |

### Using LeRobot for Collection

NVIDIA's official pipeline uses [LeRobot](https://github.com/huggingface/lerobot) for collection:

```bash
pip install lerobot

# Record episodes with SO-100 + SpaceMouse:
python -m lerobot.scripts.control_robot \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id YOUR_HF_USERNAME/my_robot_demos \
    --tags tutorial \
    --warmup-time-s 5 \
    --episode-time-s 30 \
    --reset-time-s 10 \
    --num-episodes 50 \
    --push-to-hub
```

### Data Quality Guidelines

| Rule | Why |
|---|---|
| 50+ episodes minimum | Below this, model won't generalize |
| 200+ episodes preferred | Better generalization across positions |
| Consistent task setup | Same object poses reduce variance |
| ~30% failure demos OK | Model learns recovery behaviors |
| Multiple camera angles | Cover occlusions |
| Clean, decisive motions | Noisy demos hurt more than they help |
| Vary object positions ±5cm | Prevents overfitting to single position |

---

## 7. Dataset Formatting (LeRobot Format)

GR00T fine-tuning expects **LeRobot HuggingFace dataset format**:

```
my_dataset/
├── meta/
│   ├── info.json          ← dataset metadata (fps, shapes, episode count)
│   ├── modality.json      ← obs/action schema (copy from examples/YOUR_ROBOT/)
│   └── episodes.jsonl     ← per-episode metadata
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet   ← tabular: state, action, timestamp per step
│   │   └── ...
└── videos/
    ├── chunk-000/
    │   ├── observation.image/episode_000000.mp4
    │   └── observation.wrist_image/episode_000000.mp4
    └── ...
```

### Converting Existing Data

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# If you collected with LeRobot, it's already in this format.
# If converting from ROS bags or custom format:

import torch
from pathlib import Path

# Each episode: save as parquet
import pandas as pd

episode_data = {
    "observation.state.x": [...],        # list of floats, one per step
    "observation.state.gripper": [...],  # list of arrays
    "action.x": [...],
    "timestamp": [...],
    "episode_index": [...],
    "frame_index": [...],
}
df = pd.DataFrame(episode_data)
df.to_parquet(f"data/chunk-000/episode_{ep_idx:06d}.parquet")
```

### Using a Public Dataset (fastest start)

```bash
# LIBERO Spatial (what we used for T1):
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir ./datasets/libero_spatial --repo-type dataset

# SO-100 pick-and-place:
huggingface-cli download lerobot/so100_strawberry_grape \
    --local-dir ./datasets/so100 --repo-type dataset
```

---

## 8. LoRA Fine-Tuning

### What is LoRA?

Low-Rank Adaptation — instead of updating all 3B parameters (needs 80GB+), we add small trainable adapter matrices to attention layers. Only adapters are trained (~50M params), frozen backbone provides the pre-trained knowledge.

```
Full fine-tune: update W (3B params) → needs 80GB VRAM
LoRA:           update W + A×B (50M params) → needs 40GB VRAM
                where A and B are low-rank matrices (rank=8 typically)
```

### Fine-Tuning Command

```bash
cd Isaac-GR00T

# Single GPU (L40S 48GB)
python -m gr00t.experiment.runner \
    --config-path examples/LIBERO/libero_train.yaml \
    --override \
        experiment.output_dir=./checkpoints/my_robot_ft \
        dataset.repo_id=./datasets/my_robot_demos \
        dataset.meta_path=./datasets/my_robot_demos/meta/modality.json \
        model.pretrained_model_path=./checkpoints/groot_n17_base \
        model.embodiment_tag=LIBERO_PANDA \
        training.max_steps=10000 \
        training.batch_size=32 \
        training.learning_rate=1e-4 \
        training.lora_rank=32
```

### Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `max_steps` | 10000 | More steps = better, but diminishing returns after 20k |
| `batch_size` | 32 | Reduce if OOM; use gradient accumulation |
| `learning_rate` | 1e-4 | Don't go above 5e-4 (catastrophic forgetting) |
| `lora_rank` | 32 | Higher = more capacity, more VRAM |
| `action_chunk_size` | 8 | How many steps per inference call |
| `num_cameras` | 2 | Match your modality.json |

### Multi-GPU (8× A100, paper scale)

```bash
torchrun --nproc_per_node=8 -m gr00t.experiment.runner \
    --config-path examples/LIBERO/libero_train.yaml \
    --override \
        training.max_steps=20000 \
        training.batch_size=640   # 80 per GPU × 8
```

### Expected Training Curve

```
Step    0: loss ~2.5  (random)
Step 1000: loss ~0.8
Step 5000: loss ~0.3
Step 10k:  loss ~0.15
Step 20k:  loss ~0.08  (paper-scale)
```

Training time on L40S: ~4 hrs for 10k steps.

---

## 9. Evaluation

### Simulation Evaluation

```bash
# Single task:
MUJOCO_GL=egl python -m gr00t.eval.sim.LIBERO.libero_eval \
    --checkpoint ./checkpoints/my_robot_ft/checkpoint-10000 \
    --task libero_spatial:0 \
    --num-episodes 20

# All tasks:
MUJOCO_GL=egl python -m gr00t.eval.sim.LIBERO.libero_eval \
    --checkpoint ./checkpoints/my_robot_ft/checkpoint-10000 \
    --task libero_spatial \
    --num-episodes 20
```

### Writing Your Own Eval Loop

```python
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np, math

# Load once
policy = Gr00tPolicy(
    model_path="./checkpoints/my_robot_ft/checkpoint-10000",
    embodiment_tag="LIBERO_PANDA",
    device="cuda",
)
policy.model.eval()

# Per episode:
def policy_fn(obs_dict):
    # 1. Extract state
    eef_pos = obs_dict["robot0_eef_pos"]           # (3,)
    eef_quat = obs_dict["robot0_eef_quat"]         # [w,x,y,z]
    gripper = obs_dict["robot0_gripper_qpos"]      # (2,)

    # 2. Convert quaternion → axis-angle (CRITICAL)
    quat_xyzw = [eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]
    den = np.sqrt(1.0 - eef_quat[3]**2)
    if math.isclose(den, 0.0): rpy = np.zeros(3)
    else: rpy = (eef_quat[:3] * 2.0 * math.acos(eef_quat[3])) / den

    # 3. Build observation dict
    def s(v): return np.array([[[float(v)]]], dtype=np.float32)  # (1,1,1)
    img = obs_dict["agentview_image"][::-1, ::-1]   # flip both axes!

    obs = {
        "video": {
            "image": img[None, None, ...],           # (1,1,H,W,3)
            "wrist_image": obs_dict["robot0_eye_in_hand_image"][::-1,::-1][None,None,...],
        },
        "state": {
            "x": s(eef_pos[0]), "y": s(eef_pos[1]), "z": s(eef_pos[2]),
            "roll": s(rpy[0]), "pitch": s(rpy[1]), "yaw": s(rpy[2]),
            "gripper": gripper[None, None, :],       # (1,1,2)
        },
        "language": {
            "annotation.human.action.task_description": [["pick up the bowl"]]
        },
    }

    # 4. Get action
    action_dict, _ = policy.get_action(obs)

    # 5. Assemble action array
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    action = np.concatenate([action_dict[k][0,0].flatten() for k in keys])

    # 6. Gripper transform (GR00T [0,1] → LIBERO [-1,+1] inverted)
    action[-1] = -np.sign(2.0 * action[-1] - 1.0)
    return action
```

---

## 10. Real Robot Deployment

### Architecture: Client-Server

GR00T inference is too slow to run in the same process as robot control. Use the official client-server setup:

```
┌────────────────────┐         ┌─────────────────────┐
│  Robot Controller  │  HTTP   │   GR00T Inference   │
│  (low-latency PC)  │◄───────►│   Server (GPU PC)   │
│  10-30 Hz control  │         │   ~10 Hz inference  │
└────────────────────┘         └─────────────────────┘
```

### Step 1 — Start Inference Server

```bash
# On GPU machine:
python -m gr00t.eval.service.inference_server \
    --model-path ./checkpoints/my_robot_ft/checkpoint-10000 \
    --embodiment-tag LIBERO_PANDA \
    --port 5555
```

### Step 2 — Robot Client

```python
# On robot control PC:
from gr00t.eval.service.inference_client import InferenceClient

client = InferenceClient(host="gpu-machine-ip", port=5555)

# In control loop at 10 Hz:
obs = get_obs_from_robot()   # your robot's sensor read
action_dict = client.get_action(obs)
send_to_robot(action_dict)   # your robot's controller write
```

### Step 3 — Control Loop Details

```python
import time

ACTION_CHUNK = 8      # GR00T outputs 8 steps at once
CONTROL_HZ = 10       # 10 Hz control
INFERENCE_HZ = 10 / ACTION_CHUNK  # 1.25 Hz inference

action_buffer = []
last_inference = 0

while running:
    t0 = time.time()
    obs = robot.get_obs()

    # Only call GR00T when buffer empty
    if not action_buffer:
        action_chunk = client.get_action(obs)  # returns 8 actions
        action_buffer = list(action_chunk)

    # Execute next action from buffer
    action = action_buffer.pop(0)
    robot.execute(action)

    # Sleep to maintain Hz
    elapsed = time.time() - t0
    time.sleep(max(0, 1/CONTROL_HZ - elapsed))
```

### ROS Integration

```bash
# Official ROS2 node (from Isaac-GR00T):
ros2 launch gr00t_ros groot_inference.launch.py \
    model_path:=./checkpoints/my_robot_ft/checkpoint-10000 \
    embodiment_tag:=LIBERO_PANDA

# Robot side: publish to /observations, subscribe to /actions
```

---

## 11. Using Different Base Models

### GR00T N1.5 vs N1.7

| Model | Params | Notes |
|---|---|---|
| `nvidia/GR00T-N1.5-3B` | 3B | Older, community checkpoints available |
| `nvidia/GR00T-N1.7-3B` | 3B | Latest, better pre-training, use this |
| `nvidia/GR00T-N1.7-LIBERO` | 3B | LIBERO fine-tuned, best for Franka sim |

### Other VLA Models (alternatives to GR00T)

| Model | Org | Strengths | VRAM |
|---|---|---|---|
| **GR00T N1.7** | NVIDIA | Best manipulation, active development | 16GB inference |
| **OpenVLA** | Stanford | Open weights, research-friendly | 8GB inference |
| **π0 (pi-zero)** | Physical Intelligence | Best real-robot SOTA | Not public |
| **Octo** | UC Berkeley | Small, fast, multi-task | 4GB inference |
| **RoboFlamingo** | ByteDance | Strong on LIBERO | 16GB inference |

### Switching Base Models

```python
# OpenVLA example (same pattern, different API):
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained("openvla/openvla-7b")
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",
    torch_dtype=torch.bfloat16).cuda()

inputs = processor(
    prompt="In: {<image>}\nWhat action should the robot take to pick up the bowl?\nOut:",
    image=pil_image,
    return_tensors="pt"
).to("cuda")

action = model.predict_action(**inputs, unnorm_key="bridge_orig")
```

---

## 12. Common Pitfalls & Debugging

### Pitfall 1 — Wrong Rotation Convention → 0% Success

**Symptom:** Model runs without errors but task success is always 0%.

**Cause:** Passing Euler angles instead of axis-angle, or wrong quaternion order.

```python
# WRONG: Euler angles
from scipy.spatial.transform import Rotation
euler = Rotation.from_quat([qx,qy,qz,qw]).as_euler("xyz")  # ❌

# CORRECT: axis-angle (match libero_env.py)
den = np.sqrt(1.0 - quat[3]**2)
rpy = (quat[:3] * 2.0 * np.arccos(quat[3])) / den          # ✅
```

### Pitfall 2 — Image Not Flipped → Degraded Performance

```python
img = obs["agentview_image"]             # raw from env
img_model = img[::-1, ::-1]             # flip both axes before passing to model
```

### Pitfall 3 — Gripper Sign Inversion

```python
# GR00T outputs gripper in [0,1]: 0=close, 1=open
# Many robots need: -1=close, +1=open
action[-1] = -np.sign(2.0 * action[-1] - 1.0)   # normalize → binarize → invert
```

### Pitfall 4 — EmbodimentTag Lookup

```python
EmbodimentTag("LIBERO_PANDA")    # ❌ ValueError: not valid enum value
EmbodimentTag["LIBERO_PANDA"]    # ✅ lookup by name
```

### Pitfall 5 — State Dict Shape

Each state key must be `(B=1, T=1, D)` not `(D,)`:

```python
state["x"] = np.array([[[eef_x]]], dtype=np.float32)    # (1,1,1) ✅
state["x"] = np.array([eef_x], dtype=np.float32)        # (1,) ❌
state["gripper"] = gripper[None, None, :]                 # (1,1,2) ✅
```

### Pitfall 6 — Action Dict Keys vs Array

```python
# get_action() returns a DICT keyed by joint group, NOT an array:
action_dict, _ = policy.get_action(obs)
action_dict["actions"]   # ❌ KeyError
action_dict["x"]         # ✅ shape (1, chunk_size, 1)
action_dict["gripper"]   # ✅ shape (1, chunk_size, 1)
```

### Pitfall 7 — .eval() on Wrong Object

```python
policy.eval()         # ❌ AttributeError: Gr00tPolicy has no .eval()
policy.model.eval()   # ✅ .model is the underlying nn.Module
```

### Debugging Checklist

```
[ ] modality.json keys match exactly what you pass to get_action()
[ ] Image shape is (1, 1, H, W, 3) uint8, NOT (H, W, 3) float
[ ] Images are flipped [::-1, ::-1]
[ ] Rotation is axis-angle not Euler
[ ] State shapes are (1, 1, D) not (D,)
[ ] Gripper action is inverted for your robot
[ ] EmbodimentTag accessed by name with []
[ ] policy.model.eval() called before inference
[ ] MUJOCO_GL=egl set for headless rendering
```

---

## 13. Interview Quick-Reference

**Q: What is GR00T?**
> A 3B-parameter Vision-Language-Action model pre-trained by NVIDIA on Open-X Embodiment (1M+ demos). It takes RGB images, robot state, and a natural-language task description as input, and outputs action chunks via a diffusion transformer head.

**Q: How does LoRA fine-tuning work?**
> Instead of updating all 3B parameters (needs 80GB VRAM), LoRA adds small trainable adapter matrices (rank 8-64) to the attention layers. Only ~50M parameters are trained while the backbone is frozen. This reduces VRAM to 40GB and training time to 4-8 hours.

**Q: What's the difference between state-based and image-based policies?**
> State-based uses robot proprioception (joint angles, eef position) — fast, robust, needs sensors. Image-based uses RGB cameras — more general, harder to train, needs good camera placement.

**Q: What's action chunking?**
> Instead of predicting one action per step, GR00T predicts 8 steps at once. This amortizes the expensive inference (~0.3s) over 8 control steps, achieving effective 10 Hz control with only 1.25 Hz inference.

**Q: Why axis-angle instead of Euler angles?**
> Euler angles have gimbal lock at certain orientations (90° pitch causes singularity). Axis-angle is a smooth 3D representation. GR00T's training data used axis-angle so inference must match.

**Q: How many demos do you need?**
> 50 is the minimum for a simple pick-and-place. 200-500 for generalization across object positions. 2000+ for multi-task generalization. Quality matters more than quantity — clean, decisive demos are better than noisy ones.

**Q: What is the Open-X Embodiment dataset?**
> A multi-institution robot dataset with 1M+ demonstrations across 22 robot morphologies and 527 tasks, collected from labs including Google, Stanford, Berkeley, and CMU. GR00T's pre-training on this gives it broad manipulation priors.

---

*Guide written 2026-06-06. Based on Isaac-GR00T main branch and hands-on T1 eval achieving 97% success on LIBERO Spatial (10 tasks, 200 episodes).*
