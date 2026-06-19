---
tags: [concept, rl, method]
---

# RL Method — HumanUP two-stage + AMP

The wakeboard start = **getting up under an external pull**. The closest published method is **HumanUP**, which we adapt.

## HumanUP two-stage recipe ([arxiv 2502.12152](https://arxiv.org/html/2502.12152v1))
Real getting-up on the **Unitree G1**, 78% success. Two stages:

- **Stage I — Discovery:** weak regularization, simplified collision, canonical init, fast motion allowed. Goal: *discover any feasible crouch→stand trajectory*. Rewards: pelvis-height, height-progress, uprightness, feet-contact progression, soft symmetry.
- **Stage II — Deployable:** track an **8×-slowed** Stage-I trajectory under **strict** smoothness/torque penalties, **20k randomized init poses**, full collision, domain randomization. Goal: *smooth, safe, robust, natural*.

Why two stages: getting-up has **non-periodic contacts** and **sparse reward** — discovery needs freedom; deployment needs discipline. Same logic applies to the wakeboard rise.

## AMP — style reward ([arxiv 2104.02180](https://arxiv.org/pdf/2104.02180))
A discriminator trained (GAIL-style) to tell "real wakeboard-stance demo" from "policy motion." Its output is a **style reward**: the policy is pushed to *look like* a real start without hand-tuning every joint. Needs only a **small** reference set ([[Environment_and_Rope_Model|see §9 data options]]). We turn it on in Stage II / as an ablation.

## Algorithm & infra
- **PPO via RSL-RL** (reuse the repo's runner config patterns).
- Massive parallel Isaac Lab envs (4k–8k) → on-policy PPO is sample-cheap & stable.
- Pull-speed **curriculum** (10→30 km/h) is mandatory — never start at 30.

## Why not pure imitation / offline?
We have no large wakeboard mocap dataset, and the *physics of the pull* matter — so we learn in sim with RL, using AMP only for **style**, not as the primary objective.

Related: [[Reward_Design]] · [[Wakeboard_Start_Biomechanics]] · [[Environment_and_Rope_Model]]
