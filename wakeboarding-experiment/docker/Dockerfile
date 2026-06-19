# Wakeboarding experiment image — ONE image used by BOTH Modal and Lightning.
# Build once, push to GHCR, pull on any machine -> fast machine switching.
#
# Strategy: bake the slow-changing deps into the image; MOUNT the code at runtime
# (docker-compose mounts the repo), so editing code never needs a rebuild.

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
