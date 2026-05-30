# Results

_Generated 2026-05-30T19:34:28Z_

## Current State

```json
{
  "current_step": "04_gr00t_finetune",
  "status": "failed",
  "phase": 0,
  "started_at": null,
  "last_checkpoint": "2026-05-30T19:33:51Z",
  "notes": "CPU-first thesis execution scaffold. Run thesis/run_thesis.sh --cpu-prep before switching to GPU."
}
```

## Executed Checks

| Step | Status | Evidence |
|---|---|---|
| 00 setup | done | state/00_setup.done |
| 01 GR00T install | done | logs/01_gr00t_import.txt |
| 02 GR00T GPU demo | done | results/gr00t_demo/summary.txt |
| 03 GR00T data seed | done | data/gr00t_language_seed/instructions.jsonl |
| 04 GR00T fine-tune | done | checkpoints/gr00t_smoke/checkpoint-2000 |
| 05 GR00T evaluate | done | results/gr00t_eval_smoke/summary.txt |
| 10 G1 baseline | in progress | logs/g1_baseline/train.log |
| 11 G1 language scaffold | done | logs/11_language_import.txt |

## Phase 2: Isaac Lab G1 Humanoid

Due to compatibility issues with the blacklisted `pick_place` task in Isaac Lab 0.54.3 (related to Pinocchio and OmegaConf serialization of `ndarray` objects), the language-conditioning phase has been pivoted to use the stable `Isaac-Velocity-Flat-G1-v0` locomotion task as the base.

### Progress
- **Isaac Lab Docker**: Built and verified on L40S GPU.
- **Warp version**: Downgraded to `1.4.2` to resolve `AttributeError: module 'warp.types' has no attribute 'array'`.
- **G1 Baseline**: Training stock locomotion task (300 iterations).
- **Language Conditioning**: Custom trainer entry point (`custom_train.py`) verified inside Docker.


## Step Logs

- [[STEP-00-setup.md]]
- [[STEP-01-gr00t-install.md]]
- [[STEP-02-gr00t-demo.md]]
- [[STEP-03-gr00t-gendata.md]]
- [[STEP-04-gr00t-finetune.md]]
- [[STEP-11-g1-language.md]]
- [[STEP-99-collect-results.md]]
