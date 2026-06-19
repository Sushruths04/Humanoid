# OPTIONAL — only needed IF the existing humanoid-isaaclab image is missing rsl-rl/deps.
# The wakeboarding task reuses the SAME humanoid-isaaclab image (same system, different task);
# normally you do NOT build a new image. Check first inside the container:
#     python -c "import rsl_rl"   # if this works, you don't need this Dockerfile at all.
# If you DO build, it stays the same image lineage (no second base) and code is mounted.

FROM ghcr.io/sushruths04/humanoid-isaaclab:latest

# rarely-changing Python deps (Isaac Sim/Lab already in the base image)
RUN pip install --no-cache-dir \
    "rsl-rl-lib>=2.0.0" pyyaml wandb tensorboard

# graphics caps so the SAME image can also render on Lightning (video step).
# Training ignores these; only the camera/video path uses them.
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/wakeboarding-experiment

# Code is NOT copied in on purpose — it is mounted at runtime (see docker-compose.yaml).
# For a fully self-contained image (e.g. for Modal without a code mount), uncomment:
# COPY . /workspace/wakeboarding-experiment
