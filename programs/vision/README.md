# P3 — Vision-conditioned navigation (scaffold)

Status: scaffold. Requires the GPU container with **working Vulkan/graphics**
(rendering), which the camera path needs. Gate before any work here:

```bash
bash programs/scripts/graphics_preflight.sh   # must list a GPU device
```

## What to build (when graphics confirmed)
1. A vision task that reuses the nav command + a head camera. The camera config
   already exists in `my_humanoid_project/tasks/g1_vla_vision_cfg.py`
   (TiledCamera + CNN runner). Combine it with `nav_command_obs` / the reward
   from `g1_command_nav_cfg.py` so the policy navigates from pixels.
2. Train small first (per the existing CNN runbook): `NUM_ENVS=128`,
   `128x128` camera, then scale.
3. Robustness via Cosmos-Transfer synthetic data (see `programs/cosmos`).

## Test command (later)
```bash
# after graphics_preflight passes
bash programs/scripts/train_eval_nav.sh Humanoid-G1-Vision-VLA-v0 128 300 16
```

## DoD
Vision-only success >= 50%; masked-camera probe confirms the policy uses pixels.
