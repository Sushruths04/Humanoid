---
tags: [results, live]
---

# Results (LIVE — update after every run)

> Per [[DOC_PROTOCOL]]: a number appears here only with a `results/*.json` or `train.log` source. Tables are empty until runs happen.

## Checkpoint ladder (PLAN §10.1)
| ID | Milestone | Accept | Reached? | Success | Ckpt path | Source |
|---|---|---|---|---|---|---|
| `ckpt_00_smoke` | pipeline reaches PPO | no crash | ⬜ | — | — | — |
| `ckpt_10_stage1_slow` | up @10 km/h | ≥60% | ⬜ | — | — | — |
| `ckpt_20_stage1_30` | up @30 km/h | ≥50% | ⬜ | — | — | — |
| `ckpt_30_stage2_deploy` | smooth+natural @30 | ≥70% | ⬜ | — | — | — |
| `ckpt_40_robust` | + domain randomization | ≥60% | ⬜ | — | — | — |

## Table A — success vs pull speed
| Pull speed | Success % | Fall % | Time-to-stand (s) | Board-angle adherence % | Source |
|---|---|---|---|---|---|
| 20 km/h | — | — | — | — | — |
| 25 km/h | — | — | — | — | — |
| 30 km/h | — | — | — | — | — |
| 35 km/h | — | — | — | — | — |

## Table B — ablations (each should hurt one metric)
| Variant | Success % | What it shows | Source |
|---|---|---|---|
| Full | — | reference | — |
| − pull-speed curriculum | — | can't reach 30 directly | — |
| − biomechanics rewards | — | gets up but wrong form | — |
| − AMP style | — | less natural stance | — |
| − domain randomization | — | brittle under perturbation | — |

## Table C — Stage I vs Stage II
| Stage | Success % | Smoothness (action accel ↓) | Energy (Στ² ↓) | Naturalness | Source |
|---|---|---|---|---|---|
| Stage I (discovery) | — | — | — | rough | — |
| Stage II (deployable) | — | — | — | natural | — |

## Headline (update when available)
> _No verified results yet — plan stage._
