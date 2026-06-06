---
tags: [reference, videos, index, results]
---

# Video Index — All Portfolio Videos

Every video in `programs/videos/`, what it shows, the result it demonstrates, and how the underlying model was trained.

---

## Navigation Track Videos

### `demo_reel.mp4` — 3-Way Side-by-Side Demo Reel

**What it shows:** Three navigation policies running simultaneously in Isaac Lab:
- Left: CommandNav (G1 humanoid walking to command-specified targets)
- Centre: ObstacleNav (G1 humanoid navigating around obstacles)
- Right: SeqNav (G1 humanoid visiting 3 waypoints in sequence)

**Result:** Composite portfolio demo — not a single task metric

**How it was trained:**

| Policy | Algorithm | GPU | Training time | Result |
|---|---|---|---|---|
| CommandNav | PPO (RSL-RL) | L4 | ~20 min, 500 iters | 94.5% success |
| ObstacleNav | PPO (RSL-RL) | L4 | ~20 min, 500 iters | 85.9% success |
| SeqNav | PPO (RSL-RL) | L4 | ~20 min, 500 iters | 80.9% full-sequence |

**Training setup:**
- Environment: Isaac Lab Manager-Based RL, G1 humanoid, 4096 parallel envs
- Observation: velocity commands + base lin/ang vel + projected gravity + joint pos/vel
- Action: 37-dim joint position targets
- Reward: progress toward target + reach bonus + alignment + upright (P0-stable)
- No cameras — pure proprioception (state-based)

---

## T1 — GR00T N1.7 Manipulation Videos

All 10 videos below show the **same pre-trained GR00T N1.7 policy** evaluated on different LIBERO Spatial tasks. Each video is one successful episode.

### How GR00T Was Trained

GR00T was **not trained by us** for inference. NVIDIA trained it in two stages:

**Stage 1 — Pre-training (NVIDIA, not us):**
- Dataset: Open-X Embodiment — 1M+ robot demonstrations, 22 robot types
- Model: Qwen3-VL (2.7B vision-language backbone) + DiT action head (0.3B)
- Hardware: Thousands of GPUs over months
- Result: General manipulation priors (grasping, placing, language grounding)

**Stage 2 — LIBERO Fine-tuning (NVIDIA, not us):**
- Dataset: LIBERO Spatial demonstrations (Franka Panda, tabletop pick-and-place)
- Method: Full fine-tune on LIBERO_PANDA embodiment
- Checkpoint: `nvidia/GR00T-N1.7-LIBERO/libero_spatial`

**What we did:**
- Wrote the eval harness (`evaluate_groot_libero.py`) to plug GR00T into LIBERO
- Fixed 3 obs/action convention bugs that caused 0% success (see [[00 - Failure Index]] F-11)
- Ran evaluation: 20 eps × 10 tasks → 97.0% mean success
- Ran video recording: 3 eps × 10 tasks → 100% mean success

**Inference setup:**
- Input: agentview RGB (256×256) + wrist RGB (256×256) + eef state (8-dim) + task language
- Output: 7-dim OSC delta action chunk (8 steps per inference call)
- Frequency: ~10 Hz control, 1.25 Hz inference (chunked)
- VRAM: ~16 GB on L4

---

### Video 1 — `pick_up_the_black_bowl_between_the_plate_and_the_r_success_ep00.mp4`

**Task:** Pick up the black bowl **between the plate and the ramekin**, place it on the plate

**Language instruction:** `"pick up the black bowl between the plate and the ramekin and place it on the plate"`

**Challenge:** Object is spatially constrained — bowl is between two other objects; model must identify the correct bowl from language

**Result:** 100% (3/3 episodes in video run; ~97% in full 20-ep eval)

---

### Video 2 — `pick_up_the_black_bowl_from_table_center_and_place_success_ep00.mp4`

**Task:** Pick up the black bowl **from the center of the table**, place it on the plate

**Language instruction:** `"pick up the black bowl from the table center and place it on the plate"`

**Challenge:** Most straightforward spatial reference — center of table, no occlusion

**Result:** 100% (3/3)

---

### Video 3 — `pick_up_the_black_bowl_in_the_top_drawer_of_the_wo_success_ep00.mp4`

**Task:** Pick up the black bowl **in the top drawer of the wooden cabinet**, place it on the plate

**Language instruction:** `"pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"`

**Challenge:** Most complex task — requires opening a drawer first, then grasping from inside. Tests articulated object manipulation.

**Result:** 100% (3/3)

---

### Video 4 — `pick_up_the_black_bowl_next_to_the_cookie_box_and__success_ep00.mp4`

**Task:** Pick up the black bowl **next to the cookie box**, place it on the plate

**Language instruction:** `"pick up the black bowl next to the cookie box and place it on the plate"`

**Challenge:** Relational spatial reasoning — must identify bowl relative to cookie box

**Result:** 100% (3/3)

---

### Video 5 — `pick_up_the_black_bowl_next_to_the_plate_and_place_success_ep00.mp4`

**Task:** Pick up the black bowl **next to the plate**, place it on the plate

**Language instruction:** `"pick up the black bowl next to the plate and place it on the plate"`

**Challenge:** Bowl and plate are adjacent — must pick up bowl and then place it ON the plate (not move plate)

**Result:** 100% (3/3)

---

### Video 6 — `pick_up_the_black_bowl_next_to_the_ramekin_and_pla_success_ep00.mp4`

**Task:** Pick up the black bowl **next to the ramekin**, place it on the plate

**Language instruction:** `"pick up the black bowl next to the ramekin and place it on the plate"`

**Challenge:** Relational reference to ramekin; requires distinguishing bowl from ramekin

**Result:** 100% (3/3)

---

### Video 7 — `pick_up_the_black_bowl_on_the_cookie_box_and_place_success_ep00.mp4`

**Task:** Pick up the black bowl **on top of the cookie box**, place it on the plate

**Language instruction:** `"pick up the black bowl on the cookie box and place it on the plate"`

**Challenge:** Bowl is elevated on the cookie box — requires reaching up, different grasp approach angle

**Result:** 100% (3/3)

---

### Video 8 — `pick_up_the_black_bowl_on_the_ramekin_and_place_it_success_ep00.mp4`

**Task:** Pick up the black bowl **on top of the ramekin**, place it on the plate

**Language instruction:** `"pick up the black bowl on the ramekin and place it on the plate"`

**Challenge:** Bowl stacked on ramekin — unstable starting position, requires careful lift

**Result:** 100% (3/3)

---

### Video 9 — `pick_up_the_black_bowl_on_the_stove_and_place_it_o_success_ep00.mp4`

**Task:** Pick up the black bowl **on the stove**, place it on the plate

**Language instruction:** `"pick up the black bowl on the stove and place it on the plate"`

**Challenge:** Object is on the stove element — different surface height, different approach

**Result:** 100% (3/3)

---

### Video 10 — `pick_up_the_black_bowl_on_the_wooden_cabinet_and_p_success_ep00.mp4`

**Task:** Pick up the black bowl **on the wooden cabinet**, place it on the plate

**Language instruction:** `"pick up the black bowl on the wooden cabinet and place it on the plate"`

**Challenge:** Object on top of a cabinet — elevated position, extended arm reach required

**Result:** 100% (3/3)

---

## Results Summary

| Video | Task Type | Success (3-ep) | Success (20-ep full eval) |
|---|---|---|---|
| demo_reel | Nav composite | — | P0: 94.5%, P1.3: 85.9%, P1.4: 80.9% |
| T1 task 0 | Between objects | 100% | ~97% |
| T1 task 1 | Table center | 100% | ~97% |
| T1 task 2 | Drawer (hard) | 100% | ~97% |
| T1 task 3 | Next to cookie box | 100% | ~97% |
| T1 task 4 | Next to plate | 100% | ~97% |
| T1 task 5 | Next to ramekin | 100% | ~97% |
| T1 task 6 | On cookie box | 100% | ~97% |
| T1 task 7 | On ramekin | 100% | ~97% |
| T1 task 8 | On stove | 100% | ~97% |
| T1 task 9 | On cabinet | 100% | ~97% |
| **T1 mean** | | **100%** | **97.0%** |

---

## Training Comparison: Nav vs Manipulation

| Property | Nav (P0-P1.4) | Manipulation (T1 GR00T) |
|---|---|---|
| Algorithm | PPO (model-free RL) | Pre-trained VLA (no RL by us) |
| Observation | State-only (proprioception) | Images + state + language |
| Action space | 37-dim joint targets | 7-dim OSC delta |
| Training data | Self-generated via RL rollouts | 1M+ human demos (Open-X) |
| Training time | ~20 min per task | Not trained by us (NVIDIA) |
| Training hardware | L4 (24GB) | 1000s of GPUs (NVIDIA) |
| Generalisation | Single task per checkpoint | 10 tasks from one checkpoint |
| Language-conditioned | Yes (command vector) | Yes (full natural language) |

---

## Related

- [[T1 - GR00T LoRA]] — full technical implementation
- [[00 - Failure Index]] — all bugs fixed, including the 3 that caused 0% success
- [[P0 - CommandNav]] / [[P1.3 - ObstacleNav]] / [[P1.4 - SeqNav]] — nav task docs
- [HuggingFace videos](https://huggingface.co/datasets/mitvho09/humanoid-g1-nav/tree/main/videos/t1_groot/) — all T1 videos online
