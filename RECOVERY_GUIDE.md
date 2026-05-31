# Humanoid Thesis: Environment Recovery & Setup Guide

This guide explains how to restore the research environment and large model weights on a new Lightning AI Studio.

## 1. Codebase Restoration
If the code is not already present, clone the repository:
```bash
git clone https://github.com/Sushruths04/Humanoid.git /home/zeus/content/Humanoid
cd /home/zeus/content/Humanoid
```

## 2. Restore Large Model Weights (GR00T)
The fine-tuned foundation model weights (8.8GB) are stored on Hugging Face.
```bash
# Install the required library
/home/zeus/miniconda3/envs/cloudspace/bin/pip install huggingface_hub

# Download the checkpoint
/home/zeus/miniconda3/envs/cloudspace/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mitvho09/GR00T-Humanoid', local_dir='/home/zeus/content/Humanoid/thesis/checkpoints/gr00t_smoke')"
```

## 3. Isaac Lab Docker Setup
Ensure you have switched the Studio to a **GPU machine** before running these.

```bash
cd /home/zeus/content/Humanoid/IsaacLab

# Build the base image (10-20 mins)
python3 docker/container.py build

# Start the container
python3 docker/container.py start

# Verify Warp version (Crucial for G1)
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p -m pip install warp-lang==1.4.2
```

## 4. Resume Phase 2 Tasks
Once the environment is ready, you can resume training or evaluation:
```bash
cd /home/zeus/content/Humanoid

# Run language-conditioned training
bash thesis/run_thesis.sh --all --from 12_g1_train_eval --no-autosave

# Run custom task
bash thesis/run_thesis.sh --all --from 20_custom_task --no-autosave
```
