---
tags: [concept, rl, ppo]
---

# PPO for Locomotion

## What PPO is
**Proximal Policy Optimization** — an on-policy actor-critic algorithm. It improves the policy with a **clipped** surrogate objective that prevents each update from moving the policy too far, which keeps training stable. The workhorse of legged-robot RL.

## The exact config used (from `g1_vla_vision_cfg.py`)
| Hyperparam | Value | Why |
|---|---|---|
| `clip_param` | 0.2 | standard PPO trust region |
| `entropy_coef` | 0.008 | a little exploration |
| `learning_rate` | 5e-4, **adaptive** | LR auto-tuned to hold KL≈`desired_kl` |
| `desired_kl` | 0.01 | target update size |
| `gamma` (γ) | 0.99 | discount |
| `lam` (λ) | 0.95 | GAE bias/variance knob |
| `num_learning_epochs` | 5 | reuse each batch 5× |
| `num_mini_batches` | 4 | minibatch SGD |
| `num_steps_per_env` | 24 | rollout horizon before update |

## Why PPO (not SAC/offline/model-based)?
- Sim throughput is enormous, so **sample efficiency doesn't matter** — on-policy is fine and **more stable** than off-policy here.
- PPO's clipping is robust to the noisy advantages you get from contact-rich locomotion.
- It's what **RSL-RL** ([[RSL_RL_Runner]]) implements with a fast GPU vectorized loop.

## Locomotion-specific notes
- **Reward** = velocity tracking + stability/upright + energy/effort penalties (Isaac Lab's stock G1 reward terms). The advanced tasks in [[Open_Questions_and_Next_Steps]] require *adding* goal/sequence/collision reward terms.
- **Episode length** is a key signal: early termination = a fall, so high mean episode length ⇒ stable gait ([[Phase2.5_Sim2Real_Robust]]).

## Study checklist
- Derive the clipped surrogate objective. Know why clipping ≠ a hard KL constraint.
- Explain GAE and what λ trades off.
- Explain the adaptive-LR / desired-KL trick RSL-RL uses.

Related: [[RSL_RL_Runner]] · [[Phase2.5_Sim2Real_Robust]] · [[Glossary]]
