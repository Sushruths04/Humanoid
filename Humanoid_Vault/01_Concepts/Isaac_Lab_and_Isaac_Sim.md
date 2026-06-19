---
tags: [concept, simulator]
---

# Isaac Sim & Isaac Lab

## What they are
- **Isaac Sim** — NVIDIA's GPU-accelerated robotics simulator (PhysX physics + Omniverse rendering). Runs thousands of robot instances in parallel on one GPU.
- **Isaac Lab** — the RL/learning framework built on Isaac Sim. Gives you standard robot tasks (incl. **Unitree G1** locomotion), the **manager-based env** API, and integration with trainers like [[RSL_RL_Runner|RSL-RL]].

## Why it matters here
This project never writes a physics engine or a robot model — it **subclasses Isaac Lab's stock G1 velocity-locomotion configs** (`G1FlatEnvCfg`, `G1RoughEnvCfg`) and adds mixins for language, markers, vision, and randomization ([[Architecture_Task_Hierarchy]]).

## Key concepts to know
- **Massive parallelism:** 8,192 envs at 100k+ steps/sec is what makes on-policy [[PPO_for_Locomotion|PPO]] practical. This throughput *is* the method.
- **Manager-based env:** observations/rewards/events are built from composable `*TermCfg` objects. Adding `language_command` as an `ObsTerm` is a one-liner because of this.
- **`@configclass`:** everything is a typed, inheritable config — the backbone of the 4-level task hierarchy.
- **Versions matter:** this stack pins **Isaac Sim 5.1**, and needed **`warp-lang` downgraded to 1.4.2** inside the container — exactly the kind of version detail to record per experiment.

## Running it
Always headless in Docker on a remote GPU. Camera tasks need the **rendering kit** experience file (see [[Tiled_Camera_and_Vulkan_Rendering]]).

Related: [[RSL_RL_Runner]] · [[PPO_for_Locomotion]] · [[Project_Overview]]
