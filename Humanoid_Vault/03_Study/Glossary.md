---
tags: [study, glossary]
---

# Glossary

- **Isaac Sim / Isaac Lab** — NVIDIA's GPU physics simulator + the RL framework on top of it. See [[Isaac_Lab_and_Isaac_Sim]].
- **RSL-RL** — the on-policy ([[PPO_for_Locomotion|PPO]]) trainer used by Isaac Lab for legged robots. See [[RSL_RL_Runner]].
- **Manager-based env** — Isaac Lab env style where Observations / Rewards / Events are composed from `*TermCfg` "managers." Lets you add a term (e.g. a command obs) without rewriting the env.
- **`@configclass`** — Isaac Lab decorator turning a dataclass into a composable config. The whole task hierarchy is configclasses ([[Architecture_Task_Hierarchy]]).
- **TiledCamera** — a GPU-batched camera that renders many parallel envs efficiently; required for vision RL at scale. See [[Tiled_Camera_and_Vulkan_Rendering]].
- **Nature-CNN** — the 3-conv encoder (32/64/64 ch, 8/4/3 kernels, 4/2/1 strides) from the DQN Atari paper; used here for the pixel policy.
- **Domain randomization (DR)** — randomizing sim physics (friction, mass, motor gains) so the policy transfers to reality. See [[Domain_Randomization_and_Sim2Real]].
- **GR00T** — NVIDIA's humanoid foundation model (`GR00T-N1.7-3B`) fine-tuned in Phase 1. See [[GR00T_Foundation_Model]].
- **VLA (Vision-Language-Action)** — a policy that maps camera + instruction → action. Here: Action real, Vision smoke-only, Language a placeholder. See [[Vision_Language_Action_Models]].
- **GAE (λ)** — Generalized Advantage Estimation; `lam=0.95` in the PPO config. Bias/variance knob for advantages.
- **desired_kl / adaptive schedule** — RSL-RL adapts the learning rate to keep policy-update KL near `0.01`.
- **Privileged / teacher–student** — train a teacher on ground-truth state, distill into a student that only sees pixels; the standard recipe for vision locomotion. See [[Open_Questions_and_Next_Steps]].
