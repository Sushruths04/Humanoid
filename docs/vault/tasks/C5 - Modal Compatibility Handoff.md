---
tags: [c5, modal, isaac-sim, compatibility, handoff]
---

# C5 Modal Compatibility Handoff

**Date**: 2026-06-10  
**Modal workspace**: `mitvho09`  
**Repository branch**: `feat/planned-scripts`  
**Outcome**: Modal provides the required L40S CUDA GPU, but the current Modal
runtime does not expose the NVIDIA Vulkan interface required by Isaac Sim.

> **2026-06-10 UPDATE — definitive root cause found.** The original report below
> attributed the Vulkan failure to a missing ICD manifest. That was only the
> first of several layers. A dedicated minimal Vulkan probe
> (`programs/c5_capstone/modal_vulkan_probe.py`) peeled them all back and found
> the true, unfixable cause: **Modal runs functions under the gVisor sandbox,
> whose NVIDIA passthrough (`nvproxy`) serves the CUDA-compute ioctl surface but
> not the graphics/Vulkan/modeset surface.** See "Definitive Root Cause (Vulkan
> Probe)" section below. Conclusion is unchanged — Isaac rendering cannot run on
> Modal — but it is now proven fundamental, not a config gap.

## Objective

Determine whether Modal can host the C5 loco-manipulation capstone, especially
the Isaac Sim / Isaac Lab checkpoints on an L40S 48 GB GPU.

This work did not implement or run C5. It only built and executed an Isaac
compatibility gate.

## Local Setup Completed

1. Created workspace-local Python environment: `.venv-modal`.
2. Installed Modal CLI `1.5.0`.
3. Authenticated Modal CLI to workspace `mitvho09`.
4. User explicitly accepted the NVIDIA Omniverse EULA.
5. Added `.venv-modal/` and local Modal logs to `.gitignore`.
6. Created:
   - `programs/c5_capstone/modal_isaac_smoke.py`
   - `programs/c5_capstone/MODAL_SETUP.md`

No credentials were committed to the repository. Modal wrote its token to the
normal user profile file at `C:\Users\sushr\.modal.toml`.

## Attempt 1: Existing GHCR Image

Image:

```text
ghcr.io/sushruths04/humanoid-isaaclab:latest
```

The image is approximately 7.85 GB compressed and contains one 6.99 GB layer.
Modal's image importer remained at `Copying blob` without byte-level progress.
Two attempts were stopped cleanly before any GPU task started.

Stopped app IDs:

```text
ap-9WZHjhhmQOfDF8MTDtFRMH
ap-88rXIIs4TVxWvxbzY4oayX
```

## Attempt 2: Supported Python Installation

Replaced the GHCR import with a package-built Modal image:

```text
Base: nvidia/cuda:12.8.1-runtime-ubuntu22.04
Python: 3.11
Isaac Sim: isaacsim[all,extscache]==5.1.0
Isaac Lab: v2.3.2
RL backend: rsl_rl
```

The first build failed only because Isaac Lab requested interactive Omniverse
EULA confirmation. After the user accepted the EULA, these variables were
added during image construction:

```text
ACCEPT_EULA=Y
OMNI_KIT_ACCEPT_EULA=YES
PRIVACY_CONSENT=Y
```

The image then built successfully:

```text
Built image im-1SbXCbcRWmQcei883jh26V in 436.56s
```

Modal required at least `524288 MiB` ephemeral disk for this image. The function
request was changed from 80 GiB to the minimum 512 GiB.

## Runtime Test Result

Final app:

```text
ap-a1NSVyutDbEzkI9sruHskD
```

The L40S function started and immediately ran three checks.

### GPU: Passed

```text
NVIDIA L40S, driver 580.95.05, 46068 MiB
```

CUDA compute and the required GPU capacity are available.

### Vulkan: Failed

```text
ERROR: loader_get_json: Failed to open JSON file
/etc/vulkan/icd.d/nvidia_icd.json

Cannot create Vulkan instance.
vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
```

This is the decisive blocker. Modal exposed the CUDA GPU but did not expose the
NVIDIA Vulkan ICD required by Isaac Sim's RTX renderer.

The script had set:

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

The referenced NVIDIA Vulkan file was not mounted in the Modal runtime.

### Isaac Lab: Failed

```text
ModuleNotFoundError: No module named 'isaaclab.app'
```

The source tree does contain `isaaclab/app`, so this appears to be a Python
editable-install or path issue and is probably repairable. It is not the main
blocker because Isaac Sim still cannot initialize Vulkan.

Possible follow-up for this secondary issue:

```bash
python -m pip show isaaclab
python -c "import isaaclab; print(isaaclab.__file__)"
python -c "import sys; print('\n'.join(sys.path))"
python -m pip install -e /opt/IsaacLab/source/isaaclab
```

## Definitive Root Cause (Vulkan Probe, 2026-06-10)

A minimal probe image (`nvidia/cuda` + `vulkan-tools`, builds in ~15 s) was used
to isolate the Vulkan failure cheaply instead of rebuilding the 512 GiB Isaac
image. Four iterations peeled back the layers:

1. **Missing ICD manifest** — the original gate set `VK_ICD_FILENAMES` but never
   created `/etc/vulkan/icd.d/nvidia_icd.json`. Probe wrote it at runtime.
   → next error appeared.
2. **Missing X11 dep** — `libGLX_nvidia.so.0` failed to dlopen `libXext.so.6`.
   Added `libxext6` + related X11 libs. → next error appeared.
3. **Loader version** — suspected the Ubuntu 22.04 loader (1.3.204) was too old
   for driver 580. Switched base to Ubuntu 24.04 (loader 1.3.275, ICD interface
   v7). → SAME error, so loader was not the cause.
4. **Sandbox detection** — confirmed the real cause.

### Confirmed facts on Modal L40S

- GPU/CUDA: `NVIDIA L40S, driver 580.95.05, 46 GB` — works.
- `NVIDIA_DRIVER_CAPABILITIES=all` (graphics IS requested).
- Full RTX/Vulkan driver libraries ARE mounted at runtime:
  `libGLX_nvidia.so.0`, `libnvidia-glvkspirv.so`, `libnvidia-rtcore.so`,
  `libEGL_nvidia.so.0`, `libnvidia-glcore.so`, etc.
- `nm -D libGLX_nvidia.so.0` exports the ICD entrypoints correctly:
  `vk_icdGetInstanceProcAddr`, `vk_icdGetPhysicalDeviceProcAddr`,
  `vk_icdNegotiateLoaderICDInterfaceVersion`.
- Yet the loader reports: `Could not get 'vkCreateInstance' via
  'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0` → `vkCreateInstance
  failed with ERROR_INCOMPATIBLE_DRIVER`.

### The decisive evidence

```text
dmesg:            Starting gVisor...
uname -a:         Linux modal 4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016 x86_64
/dev nodes:       /dev/nvidia-uvm, /dev/nvidia3, /dev/nvidiactl
                  (NO /dev/nvidia-modeset — the graphics device node)
```

Modal runs every function inside **gVisor**. gVisor's `nvproxy` implements the
CUDA-compute ioctl surface (which is why CUDA, GR00T, and Cosmos work) but does
not implement the graphics/Vulkan/modeset ioctl surface. The NVIDIA Vulkan ICD
loads and negotiates, but cannot create a device because those ioctls never
reach the host kernel driver, and `/dev/nvidia-modeset` is absent.

**This is a Modal runtime architecture limitation, not an image misconfiguration.
No `modal.Image` change, env var, or driver-capability flag can enable it.** The
only escape would be a Modal-provided non-gVisor / graphics-enabled runtime,
which is not available for standard functions today.

Probe script kept at `programs/c5_capstone/modal_vulkan_probe.py` for re-test if
Modal ever ships a graphics runtime. All probe apps stopped automatically; no
lingering GPU billing.

## Current Resource State

At handoff time:

```text
Active Modal apps: 0
Active Modal containers: 0
Active GPU tasks: 0
```

All compatibility apps stopped automatically or were explicitly stopped.
There is no continuing GPU billing from these tests.

## Honest Feasibility Assessment

### Suitable on Modal

- C5 skill-router development and tests.
- CPU orchestration code.
- CUDA-only GR00T inference, subject to dependency and checkpoint validation.
- CUDA-only Cosmos inference, subject to memory testing.
- Video processing and artifact generation.

### Not currently suitable on Modal

- Isaac Sim headless rendering.
- Isaac Lab environments using cameras or RTX sensors.
- CPC5.1 unified Isaac environment.
- CPC5.3 end-to-end simulator controller.
- CPC5.5 simulator evaluation and rendered videos.

These require a provider that exposes NVIDIA Vulkan/EGL graphics devices, such
as the existing Lightning AI setup or another verified RTX-capable VM.

## Questions for the Next Agent

1. Does Modal now offer an officially documented Vulkan/graphics-enabled GPU
   runtime or a special configuration not used here?
2. Can Modal mount the NVIDIA Vulkan ICD and required driver libraries into a
   function container?
3. If not, should C5 use a split architecture:
   Modal for GR00T/Cosmos services and Lightning AI for Isaac simulation?
4. Is the network latency of a split architecture acceptable, or should all
   runtime components remain on one Lightning AI L40S machine?
5. Can the `isaaclab.app` import issue be repaired independently for
   non-rendering tests?

## Recommended Next Step

Use Lightning AI L40S for the complete staged C5 baseline. Keep Modal only as an
optional CUDA service after the unified Isaac environment and staged controller
work locally on Lightning.

Do not begin with the Cosmos veto. First validate:

1. Unified G1 environment.
2. P3 navigation policy loading.
3. T1 manipulation policy loading.
4. Staged navigation-to-manipulation hand-off.
5. Only then add occasional Cosmos lookahead.

