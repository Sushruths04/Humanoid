---
tags: [concept, environment, physics]
---

# Environment & Rope Model

Full spec in `../PLAN.md` §3–§4. Summary:

## The world (dry analog of water)
- **Robot:** Unitree G1 with **arms actuated** (must hold the handle — unlike the locomotion tasks that freeze arms).
- **Board:** thin rigid box (~1.4 m × 0.4 m, ~3 kg) **welded to the feet** (rigid bindings).
- **Sand surface:** frictional plane with **moderate-low friction** (μ ≈ 0.3–0.5) so the board *slides* forward under pull — the "sand instead of water" abstraction (board glides on ground instead of planing on water).

## The rope (the 30 km/h pull)
Two models behind a config flag:
- **A — constant force:** fixed horizontal force at the handle. Simple, uncapped speed.
- **B — velocity-target spring (default):** a virtual anchor moves forward at `v_pull`; a PD/spring force pulls the handle toward it, capped at `F_max`. Naturally yields "boat tows you to 30 km/h."
- **`v_pull` = 30 km/h = 8.33 m/s** along +x; ramped by the curriculum.

## Initial pose (the cannonball)
Hips/knees deeply flexed (knees to chest), torso reclined ~30–45°, board in front tipped up ~15°, arms straight on the handle near the hips. Randomized per [[Reward_Design|DR]].

## Observations (angle-centric — chosen to enable the Phase-2 CV coach)
Proprioception + base state + **board pitch/velocity** + **rope force & handle-rel-pelvis** + **`v_pull`** + a **phase clock**. Joint-angle-centric obs make the later human-coaching comparison straightforward (`PLAN.md` §11).

## Termination
Fall (torso/head/knee ground contact), feet leave board, board pitch out of range, or torso tips over.

Related: [[RL_Method_HumanUP_AMP]] · [[Reward_Design]] · [[Wakeboard_Start_Biomechanics]]
