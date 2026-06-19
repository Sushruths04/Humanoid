---
tags: [project, pipeline, scripts]
---

# Pipeline & Scripts

The whole thesis is automated under `thesis/scripts/`. Each numbered step writes a **`thesis/state/<step>.done`** marker and a **`thesis/PROGRESS/step-logs/STEP-*.md`** log — so progress is resumable and auditable across machine switches.

## The numbered pipeline
| # | Script | Does |
|---|---|---|
| 00 | `00_setup.sh` | env / container bootstrap |
| 01 | `01_gr00t_install.sh` | install GR00T deps |
| 02 | `02_gr00t_demo.sh` | preflight + demo rollout |
| 03 | `03_gr00t_gendata.sh` | generate fine-tune data |
| 04 | `04_gr00t_finetune.sh` | 10k-step fine-tune → `checkpoint-10000` |
| 05 | `05_gr00t_eval.sh` | action MSE/MAE eval |
| 10 | `10_g1_baseline_train.sh` | stock G1 flat locomotion |
| 11 | `11_g1_language_cond.sh` | + 16-dim command embedding |
| 12 | `12_g1_train_eval.sh` | train + eval G1 |
| 20 | `20_custom_task.sh` | MarkerNav (red/blue spheres) |
| 25 | `25_robust_training.sh` | rough terrain + domain randomization |
| 30 | `30_vision_vla.sh` | camera smoke (Vulkan path) |
| 31 | `31_vision_vla_cnn.sh` | 128×128 CNN vision training |
| 32 | `32_vision_vla_play.sh` | record mp4 rollout |
| 99 | `99_collect_results.sh` | aggregate summaries into `results/` |

## Support scripts
- `lib.sh` — shared helpers. `docker_image_portability.sh` — save/load/push/pull the Isaac Lab image (GHCR or tarball fallback).
- `machine_switch.sh`, `bootstrap_remote_machine.sh`, `autosave.sh`, `checkpoint.sh`, `sync_phase3_artifacts.sh` — the [[Lightning_AI_Workflow|machine-switch / backup]] machinery.

## Completion state (real `.done` markers)
`00, 01, 02, 03, 04, 05, 10, 11, 12, 20, 30, 99` are marked done.
⚠️ Note an inconsistency: **`25_robust_training` has no `.done` marker**, yet the robust checkpoint + log clearly exist — so the state markers slightly under-report what ran. Trust the artifacts in `checkpoints/`/`logs/` over the markers.

Related: [[Phase1_GR00T]] · [[Phase2.5_Sim2Real_Robust]] · [[Results_Summary]]
