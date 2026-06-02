# GPU / VRAM Requirements Per Task (rent the right size, not the biggest)

**Created:** 2026-06-02 · Companion to `MASTER_ROADMAP_CONVERGED.md`.

**TL;DR:** ~70% of this program is **16–24GB** work. Rent a **24GB** instance (Lightning L4 / RTX 4090 / A10) as your default. Step up to **48GB (L40S)** only for *scaled vision rendering*, and to **80GB (A100-80)** only for the two *Cosmos post-training* runs. Do **not** pay for 48–80GB to do 15GB of work.

---

## Two hard rules before you pick a GPU

1. **Rendering vs. compute is the deciding factor.**
   - **Isaac Sim camera rendering (vision phases P3/T3/C5)** needs an **RTX / RT-core GPU**: **L4, L40S, RTX 4090/A6000, A40**. 
   - **A100 / H100 are compute-only (no RT cores / no display engine)** — great for Cosmos training, GR00T LoRA, and world models, but **poor/unreliable for Isaac Sim cameras**. Don't rent an A100 for vision RL.
2. **VRAM scales with what you control:** env count (Isaac Lab), camera resolution, and batch size. Every number below assumes the efficiency rules (bf16, gradient checkpointing, LoRA, 8-bit optimizer). You can almost always drop VRAM by lowering env count / resolution / batch.

---

## Per-task VRAM table

| Task (checkpoints) | Min VRAM | Comfortable | Rent this | Needs RT cores? |
|---|---:|---:|---|:---:|
| **P0** honest G1 nav baseline (state RL) | 12GB (≤1k envs) | 24GB (4k envs) | **24GB** L4/4090/A10 | RTX-class (sim) |
| **P1** language-conditioned nav (state RL) | 12GB | 24GB | **24GB** L4/4090 | RTX-class (sim) |
| **P2 / T2** small world model from scratch (pure PyTorch, no Isaac Sim) | **8GB** | 12–16GB | **16GB** (T4/L4/local) | No — any CUDA GPU |
| **T0** manipulation env + BC baseline (Franka/LIBERO) | 12GB | 24GB | **24GB** L4 | RTX-class (sim) |
| **T1** GR00T N1.7 LoRA language manip (no rendering) | 16GB | 24GB | **24GB** L4/A10/A100-40 | No |
| **T2** Diffusion Policy / ACT imitation | 12GB | 24GB | **24GB** L4 | No (state) / RTX (if image obs) |
| **P3 / T3** vision RL (camera per env) — **smoke** (128 envs, 128²) | 24GB | 32GB | **24GB** L4 | **Yes — L4/L40S** |
| **P3 / T3** vision RL — **scaled** (more envs / higher res) | 40GB | 48GB | **48GB** L40S | **Yes — L40S** |
| **Cosmos Transfer 2.5** 2B inference (synthetic data gen) | 24GB (offload) | 40GB | **40GB** A100-40 / L40S | No |
| **Cosmos Predict 2.5** 2B inference (CP4.1 / CPT4.1) | 24GB (offload) | 40GB | **40GB** A100-40 / L40S | No |
| **Cosmos Predict 2.5** 2B **LoRA post-train** (CP4.3 / CPT4.3) ⚠️ | 40GB | **80GB** | **80GB** A100-80 (burst only) | No |
| **Cosmos Reason 2** 2B inference (C5 reasoning) | 8GB | 16GB | **16GB** L4/T4 | No |
| **C5** capstone integration (env + 2B inference models in loop) | 24GB | 48GB | **48GB** L40S | **Yes — L40S** |

⚠️ = the only genuine 80GB need in the whole program. Everything else is ≤48GB, and most is ≤24GB.

---

## What to actually rent, by phase (Lightning AI tiers)

| Phase | Default rental | Why | Burst (only if needed) |
|---|---|---|---|
| Ph0 (P0+T0) | **L4 24GB** | state RL + env setup | — |
| Ph1 (P1+T1) | **L4 24GB** | state RL + GR00T LoRA | — |
| Ph2 (P2+T2) | **L4 24GB** or even local | world model is tiny; no Isaac Sim for P2 | — |
| Ph3 (P3+T3) | **L40S 48GB** | scaled vision rendering needs RT cores + VRAM | L4 24GB for smoke tests |
| Ph4 (P4+T4) | **L4/L40S** for inference & data-gen | most of the phase is generation/eval | **A100-80GB for the two LoRA post-trains only** |
| Ph5 (C5) | **L40S 48GB** | vision env + Cosmos-in-loop | — |

**Cost intuition (on-demand, approximate):** 24GB ≈ $0.5–0.9/hr · 48GB (L40S) ≈ $1.0–2.0/hr · 80GB (A100) ≈ $2–3.5/hr. So: run the cheap 24GB tier for Ph0–Ph2 and all smoke tests, only pay L40S during Ph3/Ph5, and only touch A100-80 for the handful of days of Cosmos post-training in Ph4.

---

## Rules of thumb to stay cheap
- **Smoke-test on 24GB**, scale on a bigger card only after the pipeline is proven.
- **If you hit OOM, cut env count or camera resolution first** before renting up — vision VRAM is roughly linear in (num_envs × H × W).
- **World-model and GR00T-LoRA work never needs rendering** → use a cheap compute GPU (or A100-40 if idle/credited), not an L40S.
- **Reserve A100-80GB for Ph4 post-training bursts only.** Generate Cosmos synthetic data once, cache on Hugging Face, and you won't need the big card again.
- **A100 ≠ good for cameras.** If a vision run behaves strangely on an A100, that's expected — move it to L4/L40S.
