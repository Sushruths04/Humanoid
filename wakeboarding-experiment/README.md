# 🏄 Wakeboarding Start — Humanoid RL

Train a **Unitree G1** in **Isaac Lab** to perform a wakeboard **deep-water start** in a **dry/sand** analog: begin crouched & seated, feet on a board, arms straight on a rope handle; a rope ramps the pull to **30 km/h**; learn the correct biomechanics to rise to a stable ride **without face-planting**.

- **Full plan (executor-ready):** [`PLAN.md`](./PLAN.md)
- **Documentation rules:** [`DOC_PROTOCOL.md`](./DOC_PROTOCOL.md)
- **Live knowledge base (Obsidian):** open [`vault/`](./vault/) → start at `00_INDEX.md`

## Status
🟡 **PLAN written, no code yet.** Next: scaffold + board/rope/env (PLAN §14 task list).

## TL;DR of the approach
- **Method:** PPO (RSL-RL) + **HumanUP** two-stage (discovery→deployable) + optional **AMP** style reward.
- **The hard part:** the 30 km/h yank — handled by a **pull-speed curriculum** (10→30 km/h).
- **Reward = correct wakeboard form:** positive board angle, straight arms, handle at hips, gradual leg extension, don't pull against the rope.
- **Deliverables for demos:** success-rate-vs-speed table, ablations, Stage I vs II, rollout video — all kept live in `vault/03_Results/`.

## Relation to the rest of the repo
A **separate sub-project** inside the Humanoid repo. Reuses the same Isaac Lab + RSL-RL + Lightning AI + Docker stack. Distinct from the G1 locomotion/vision tasks under `my-humanoid-project/`.
