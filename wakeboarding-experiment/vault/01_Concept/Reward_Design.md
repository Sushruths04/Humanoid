---
tags: [concept, reward]
---

# Reward Design

Authoritative weights live in `../PLAN.md` §5. This note explains the *intent* so you can tune sanely. Reward = **task** + **biomechanics** + **style** − **penalties**, each an independent Isaac Lab `RewTerm` (so every term is ablatable).

## Task (get up & stay up) — from [[RL_Method_HumanUP_AMP|HumanUP]]
- **pelvis_height / height_progress** — rise toward a standing-ride height; progress is **phase-gated** (only rewarded after t>0.5 s) so the robot doesn't lunge up instantly.
- **uprightness** — torso gravity-z projection.
- **survival / forward_glide** — stay alive; board tracks `v_pull`.
- **success_bonus** — sparse +50 when the [[Wakeboard_Start_Biomechanics|stable-ride]] criteria hold 1.5 s.

## Biomechanics shaping (make it a *correct* start) — from the [[Wakeboard_Start_Biomechanics|5 rules]]
- **board_positive_angle** → rule 2 (≥10° heel-weighted plane-up).
- **arms_straight** + **handle_at_hips** → rule 3.
- **lean_back_moderate** → rule 5 (and it's *penalized* if back-lean co-occurs with elbow flexion = pulling the rope).
- **knee_bend_maintained** → rules 1 & 5 (soft knees, gradual extension via a relaxing band).

## Style (optional) — [[RL_Method_HumanUP_AMP|AMP]]
- Discriminator-based **style reward** so the motion *looks* like a real start. On in Stage II / as an ablation.

## Penalties — from HumanUP regularization
- **stand_too_fast** (rule 1), **pull_against_rope** (rule 4), torque/energy, action smoothness, dof limits, **fall** (terminal −20).

## Tuning order (when it misbehaves)
1. If it never gets up → raise `height_progress`/`success_bonus`, soften penalties (Stage-I mode).
2. If it gets up but **wrong form** → raise biomechanics terms + turn on AMP.
3. If it's twitchy/unsafe → raise smoothness/torque penalties (Stage-II mode).
4. If it cheats (reward hacking) → inspect rollouts, add/raise the specific biomechanics term it's violating.

Related: [[Wakeboard_Start_Biomechanics]] · [[RL_Method_HumanUP_AMP]] · [[Results_Live]]
