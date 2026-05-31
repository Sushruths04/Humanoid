# Future Research Roadmap: Humanoid VLA

This document outlines the logical next steps for extending the current G1 Humanoid thesis work into a more advanced research project.

## 1. Vision Integration (The "V" in VLA)
**Goal**: Transition from "Blind" navigation to "Visual" navigation.
- **Task**: Enable the front-facing camera on the Unitree G1.
- **Implementation**: Integrate a visual encoder (e.g., a lightweight ResNet or Vision Transformer) into the RSL-RL policy.
- **Challenge**: Requires rendered observations, which will increase VRAM usage and decrease training speed, necessitating high-performance cards like the L40S.

## 2. Sim-to-Real Transfer (Domain Randomization)
**Goal**: Make the policy robust enough for physical robot deployment.
- **Task**: Implement Extensive Domain Randomization (EDR).
- **Implementation**: Randomize friction, limb masses, motor gains, and add external force disturbances (pushes) during training.
- **Outcome**: A policy that can be deployed on the real Unitree G1 hardware.

## 3. Loco-Manipulation (Legs + Arms)
**Goal**: Perform complex tasks while walking.
- **Task**: Command "Pick up the object at the red marker."
- **Implementation**: Unfreeze the G1's arm and hand joints. Use a unified policy to coordinate walking and reaching/grasping simultaneously.
- **Research Question**: Can the language embedding effectively gate both locomotion and manipulation sub-tasks?

---
*Created by Gemini CLI Agent for Sushruth (May 2026)*
