# Modal C5 Compatibility Gate

This gate verifies Modal L40S compatibility with Isaac Sim 5.1 and Isaac Lab
2.3.2. It does not run C5, download checkpoints, or start training.

## One-time local setup

From the repository root in PowerShell:

```powershell
python -m venv .venv-modal
.\.venv-modal\Scripts\python.exe -m pip install --upgrade pip modal
.\.venv-modal\Scripts\modal.exe setup
```

The final command opens Modal authentication. Do not put Modal, Hugging Face,
GitHub, or registry tokens in this repository.

## Run the compatibility gate

```powershell
.\.venv-modal\Scripts\modal.exe run programs\c5_capstone\modal_isaac_smoke.py
```

The image is built from `nvidia/cuda:12.8.1-runtime-ubuntu22.04` using NVIDIA's
supported Python installation:

```text
isaacsim[all,extscache]==5.1.0
IsaacLab v2.3.2
```

The first build is large and can take a while, but Modal caches the individual
build steps. The function requests one L40S, 8 CPU cores, 32 GiB RAM, and Modal's
minimum 512 GiB ephemeral disk allocation for this image. It is single-use and
has a 20-minute execution timeout.

Proceed with C5 only if all of these pass:

1. `nvidia-smi` reports an L40S with 48 GB VRAM.
2. Vulkan initializes, or the image lacks `vulkaninfo` but the next check passes.
3. Isaac Sim compatibility checker passes when present.
4. Isaac Lab `create_empty.py --headless` exits successfully.

If the headless boot fails due to Vulkan/EGL/RTX device access, Modal is not a
valid host for the Isaac-based C5 checkpoints. Do not spend time adapting C5
until this gate passes.
