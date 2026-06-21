"""Run the wakeboard training on Modal serverless GPU (modal.com).

Why this works on Modal: training is COMPUTE-only (no cameras), so it avoids the Vulkan
graphics path. Use an RT-core GPU that Isaac Sim supports — **L40S** (or L4). Do NOT pick
A100/H100: Isaac Sim needs RT cores and won't render/initialize on them.

Setup (once):
    pip install modal && modal token new
    modal volume create wakeboard-ckpts
    # build/pull the Isaac Lab image (see image below)

Train Stage I:
    modal run modal_app.py --action train --config configs/stage1.yaml
Eval:
    modal run modal_app.py --action eval --checkpoint /ckpts/wakeboard_stage1/model_latest.pt

Notes:
- Modal function timeout maxes at 24h; long Stage-II runs checkpoint to the Volume and are
  resumed with --resume (the loop already saves model_<iter>.pt each window).
- The Isaac Sim base image is large (~20GB); first cold start is slow, then cached.
"""
from __future__ import annotations

from pathlib import Path

import modal

# Repo code must live INSIDE the image — unlike the local docker-compose path (which mounts
# the code), Modal has no mount, so we copy this experiment dir to /workspace/wakeboarding-experiment
# (the same path train()/evaluate() cd into below).
_LOCAL_DIR = Path(__file__).parent.resolve()
_REMOTE_DIR = "/workspace/wakeboarding-experiment"

# --- container image: Isaac Sim 5.1 base + Isaac Lab + this repo's deps + this repo's code ---
# Reuse YOUR existing GHCR image that already has Isaac Lab:
#   ghcr.io/sushruths04/humanoid-isaaclab:latest
image = (
    modal.Image.from_registry(
        "ghcr.io/sushruths04/humanoid-isaaclab:latest",
        add_python="3.10",
    )
    .entrypoint([])
    .run_commands("apt-get update -qq && apt-get install -y -q ffmpeg || true")
    .pip_install("rsl-rl-lib>=2.0.0", "pyyaml", "wandb", "tensorboard", "Pillow")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES",
          "CUDA_LAUNCH_BLOCKING": "1"})
    # bake the experiment code into the image so cwd=_REMOTE_DIR exists at runtime
    .add_local_dir(str(_LOCAL_DIR), _REMOTE_DIR, copy=True,
                   ignore=["runs", "checkpoints", "__pycache__", "*.pt", "vault"])
)

app = modal.App("wakeboard-rl", image=image)
ckpts = modal.Volume.from_name("wakeboard-ckpts", create_if_missing=True)

GPU = "L40S"   # RT-core GPU Isaac Sim supports. Alternatives: "L4" (cheaper), "A10G".


@app.function(gpu=GPU, volumes={"/ckpts": ckpts}, timeout=24 * 60 * 60)
def train(config: str, num_envs: int | None = None, max_iterations: int | None = None,
          resume: str | None = None):
    import subprocess
    cmd = ["python", "train.py", "--config", config, "--headless",
           "--experiment_dir", "/ckpts"]
    if num_envs:
        cmd += ["--num_envs", str(num_envs)]
    if max_iterations:
        cmd += ["--max_iterations", str(max_iterations)]
    if resume:
        cmd += ["--resume", resume]
    shell_cmd = (
        "ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && "
        "export ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps "
        "CARB_APP_PATH=/isaac-sim/kit LD_PRELOAD=/isaac-sim/kit/libcarb.so "
        "RESOURCE_NAME=IsaacSim && "
        "source /isaac-sim/setup_python_env.sh && "
        + " ".join(cmd)
    )
    subprocess.run(["bash", "-c", shell_cmd], check=True, cwd=_REMOTE_DIR)
    ckpts.commit()


@app.function(gpu=GPU, volumes={"/ckpts": ckpts}, timeout=2 * 60 * 60)
def evaluate(checkpoint: str, v_pull_kmh: float = 30.0, episodes: int = 200):
    import subprocess
    out = f"/ckpts/results/eval_{int(v_pull_kmh)}kmh.json"
    cmd = ["python", "eval.py", "--checkpoint", checkpoint,
           "--v_pull_kmh", str(v_pull_kmh), "--episodes", str(episodes),
           "--out", out]
    shell_cmd = (
        "ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && "
        "export ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps "
        "CARB_APP_PATH=/isaac-sim/kit LD_PRELOAD=/isaac-sim/kit/libcarb.so "
        "RESOURCE_NAME=IsaacSim && "
        "source /isaac-sim/setup_python_env.sh && "
        + " ".join(cmd)
    )
    subprocess.run(["bash", "-c", shell_cmd], check=True, cwd=_REMOTE_DIR)
    ckpts.commit()


@app.function(gpu=GPU, volumes={"/ckpts": ckpts}, timeout=60 * 60)
def render_video(checkpoint: str, v_pull_kmh: float = 10.0, episodes: int = 3, steps: int = 400):
    import subprocess
    out = f"/ckpts/videos/rollout_{checkpoint.split('/')[-1].replace('.pt','')}.mp4"
    cmd = ["python", "play.py", "--checkpoint", checkpoint,
           "--v_pull_kmh", str(v_pull_kmh), "--episodes", str(episodes),
           "--steps", str(steps), "--out", out]
    shell_cmd = (
        "ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && "
        "export ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps "
        "CARB_APP_PATH=/isaac-sim/kit LD_PRELOAD=/isaac-sim/kit/libcarb.so "
        "RESOURCE_NAME=IsaacSim && "
        "source /isaac-sim/setup_python_env.sh && "
        + " ".join(cmd)
    )
    subprocess.run(["bash", "-c", shell_cmd], check=True, cwd=_REMOTE_DIR)
    ckpts.commit()
    return out


@app.local_entrypoint()
def main(action: str = "train", config: str = "configs/stage1.yaml",
         checkpoint: str = "", v_pull_kmh: float = 10.0):
    if action == "train":
        train.remote(config=config)
    elif action == "eval":
        evaluate.remote(checkpoint=checkpoint, v_pull_kmh=v_pull_kmh)
    elif action == "render":
        ckpt = checkpoint or "/ckpts/wakeboard_stage1/model_latest.pt"
        out = render_video.remote(checkpoint=ckpt, v_pull_kmh=v_pull_kmh)
        print(f"[render] video saved to Modal volume: {out}")
        print(f"[render] download with: modal volume get wakeboard-ckpts {out} ./rollout.mp4")
