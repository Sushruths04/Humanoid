---
tags: [concept, sim2real, robustness]
---

# Domain Randomization & Sim-to-Real

## The problem
A policy trained in *one* exact simulator will overfit its physics and fail on the real robot (the **sim-to-real gap**). Real hardware has uncertain friction, mass, actuator gains, sensor noise, and uneven ground.

## The fix: Domain Randomization (DR)
Randomize the uncertain sim parameters **every episode**. The policy can't exploit any single physics setting, so it learns a **robust** behavior that covers the real values within the randomized range.

## What this project randomizes (from `LanguageConditionedG1RobustTaskCfg`)
- **Terrain:** flat → **procedural rough terrain** (`G1RoughEnvCfg`).
- **Actuators:** joint **stiffness** and **damping** each ±25%:
  ```python
  stiffness_distribution_params = (0.75, 1.25)
  damping_distribution_params   = (0.75, 1.25)
  ```

## Result (the project's most defensible win)
Mean reward **22.82**, mean episode length **981/1000** → robot **rarely falls** on rough terrain with randomized joints. See [[Phase2.5_Sim2Real_Robust]] / [[Results_Summary]].

## Why ±25%?
A bias/variance trade-off: too narrow ⇒ no robustness; too wide ⇒ task becomes unlearnable or the policy gets over-conservative. ±25% is a common sweet spot for actuator gains.

## What's missing for true sim-to-real (next steps)
- **External force / push disturbances** (planned — `FUTURE_WORK.md` "EDR"). See [[Open_Questions_and_Next_Steps]] Step 5.
- Randomize **friction, link masses, sensor latency/noise**.
- Eventually: deploy on real **Unitree G1** hardware.

Related: [[Phase2.5_Sim2Real_Robust]] · [[PPO_for_Locomotion]] · [[Open_Questions_and_Next_Steps]]
