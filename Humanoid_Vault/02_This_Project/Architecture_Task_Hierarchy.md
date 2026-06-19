---
tags: [project, architecture]
---

# Architecture — the Task Hierarchy

The cleverest part of the codebase is a **4-level config inheritance chain**. Each level adds *one* capability, so you can train/debug at any rung. All configs are Isaac Lab `@configclass` and subclass the **stock G1 velocity-locomotion** envs.

```
G1FlatEnvCfg  (Isaac Lab stock: flat-ground velocity locomotion)
   │
   ▼ + 16-dim language command embedding (ObsTerm)
LanguageConditionedG1EnvCfg
   │
   ▼ + Red/Blue visual marker spheres in the scene
LanguageConditionedG1CustomTaskCfg     ← "MarkerNav"
   │
   ▼ + head-mounted TiledCamera + RGB obs group (for CNN policy)
G1VisionVLAEnvCfg                       ← "Vision-VLA"

(parallel branch, off the stock ROUGH env:)
G1RoughEnvCfg  →  LanguageConditionedG1RobustTaskCfg
   = language + markers + ROUGH terrain + domain randomization
```

**Source:** `my-humanoid-project/my_humanoid_project/tasks/g1_language_pickplace_cfg.py` and `.../g1_vla_vision_cfg.py`.

## Why this design is good (interview point)
- **Separation of concerns:** language, vision, and terrain are *independent* mixins. You can ablate any one.
- **CPU-importable:** the module wraps Isaac Lab imports in `try/except ISAACLAB_AVAILABLE`, with a placeholder class fallback — so you can syntax-check tasks on a laptop *without* Isaac Sim. (See the `else:` branch defining a CPU placeholder.)
- **Env-var driven:** camera size, update period, max-iters etc. are read from environment variables (`VLA_CAMERA_HEIGHT`, `VISION_VLA_MAX_ITERS`, …) so the *same* config scales from a 16-env smoke test to 2048-env production without code edits.

## The observation structure
- **`policy` group (proprioception + command):** stock G1 locomotion observations **+** a `language_command` term ([[Phase2_G1_Locomotion_and_Language]]).
- **`images` group (vision only):** normalized CHW RGB from the head camera, consumed by a CNN encoder. Defined as a *separate* obs group so the CNN and MLP paths stay clean — see `obs_groups = {"actor": ["policy", "images"], "critic": [...]}` in the runner cfg.

Related: [[RSL_RL_Runner]] · [[Phase3_Vision_VLA]] · [[Domain_Randomization_and_Sim2Real]]
