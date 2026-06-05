---
tags: [reference, glossary, definitions]
---

# Glossary

## Isaac Lab Terms

**ManagerBasedRLEnv** — Isaac Lab's base RL environment class. You subclass it and register named terms into managers (observations, rewards, events, commands). All nav tasks here use it.

**EventTerm / EventTermCfg** — a function registered with the event manager. Fires at `mode="reset"` (episode start) or `mode="interval"` (every step with `interval_range_s=(0.0, 0.0)`).

**ObsTerm / ObservationTermCfg** — a function registered with the observation manager. Its output is concatenated into the policy's observation vector.

**RewardTerm / RewardTermCfg** — a function registered with the reward manager. Returns per-env scalar; scaled by `weight`.

**G1FlatEnvCfg** — the pre-built Unitree G1 flat-terrain locomotion env. All P-track nav tasks subclass this.

**G1FlatPPORunnerCfg** — the PPO runner config for G1. Experiment name: `g1_flat`. All nav tasks share it.

**vel_command_b** — the base-frame velocity command buffer: `[vx, vy, wz]` where vx=forward, vy=sideways, wz=yaw rate. Written to by the steer event. Tracked by the locomotion policy.

**`isaac-lab-base`** — the name of the Docker container (also the image tag after re-tagging).

**`isaaclab.sh -p <script.py>`** — the entrypoint for running Python inside the Isaac Sim environment. Sets up Omniverse Kit, loads plugins, then runs your script.

---

## RL Terms

**PPO (Proximal Policy Optimization)** — the RL algorithm used. Updates the policy to maximize reward while staying "close" (KL-bounded) to the old policy. Implemented by RSL-RL.

**RSL-RL** — fast vectorized PPO library built for Isaac Lab. `OnPolicyRunner` is the main class. Checkpoint format: `.pt` PyTorch state dicts.

**Bootstrap problem** — when a policy can't reach a reward for the first time (rare event in early training), gradient-based methods can't learn from it. Fix: make the first success easier.

**Local optimum** — a policy that maximizes reward in one mode (e.g., standing still) and can't escape to a globally better mode (e.g., navigating) because the gradient points in the wrong direction.

**Progress reward** — `reward = prev_dist - cur_dist`: positive when the robot closes distance to the target. Dense signal every step.

**Reach bonus** — sparse reward (+10) when the robot arrives within `reach_radius` of the goal. Motivates actually committing to the goal.

**Potential field** — a steering technique: obstacles create repulsive "forces" that deflect the velocity command, allowing smooth obstacle avoidance without learning.

**RSSM (Recurrent State-Space Model)** — the world model architecture: combines a GRU (deterministic state) with a sampled Gaussian (stochastic latent) to model environment dynamics.

---

## Pipeline Terms

**Command-conditioned** — a policy/behavior that changes based on an explicit command input (e.g., "go to marker 0"). Contrasted with unconditional (ignores the command).

**Instruction-swap probe** — testing whether a policy's behavior actually changes when the command changes. The minimal proof that conditioning is genuine.

**DoD (Definition of Done)** — the quantitative success criterion for a checkpoint. E.g., CP1.3 DoD: `success_rate >= 65%`.

**Bind mount** — a Docker feature mapping a host directory into the container. Changes made inside the container appear on the host and persist. CRITICAL for not losing results.

**Ephemeral storage** — data that disappears when the Docker container is removed or the machine restarts (everything NOT in a bind-mounted path or Docker volume).

---

## Hardware Terms

**L4** — NVIDIA L4 GPU, 24 GB VRAM, RTX (has RT cores for rendering). Used for P0–P1 training and demo video rendering. ~$0.5–0.9/hr.

**T4** — NVIDIA T4, 16 GB VRAM, no RT cores. Sufficient for state-based nav RL (measured ~4.6 GB peak). Cheaper than L4. ~$0.3–0.5/hr.

**L40S** — 48 GB VRAM, RTX. Use for scaled vision RL (P3, P4 inference). ~$1.0–2.0/hr.

**A100-80** — 80 GB VRAM, no RT cores. Only needed for Cosmos Predict post-training (Phase 4). ~$2–3.5/hr.

**RT cores** — hardware ray-tracing cores. Required for Isaac Sim camera rendering. A100 does NOT have them — never use A100 for vision RL.

---

## Related

- [[00 - START HERE]]
- [[Isaac Lab Manager-Based RL]]
- [[PPO with RSL-RL]]
- [docs/GPU_VRAM_REQUIREMENTS.md](../../docs/GPU_VRAM_REQUIREMENTS.md)
