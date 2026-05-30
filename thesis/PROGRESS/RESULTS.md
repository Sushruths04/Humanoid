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
| 11 G1 language scaffold | done | logs/11_language_import.txt |

## GR00T Evaluation Summary (Smoke)

```text
# GR00T eval smoke summary

Generated: 2026-05-30T20:49:05Z
Exit status: 0
Checkpoint: .../checkpoint-2000
Average MSE: 25.87
Average MAE: 3.01
```

## Next Milestone: Isaac Lab G1 Baseline

Phase 1 (GR00T replication) is substantially complete with the successful fine-tune and open-loop evaluation. The next phase involves setting up the G1 humanoid baseline in Isaac Lab.


## Step Logs

- [[STEP-00-setup.md]]
- [[STEP-01-gr00t-install.md]]
- [[STEP-02-gr00t-demo.md]]
- [[STEP-03-gr00t-gendata.md]]
- [[STEP-04-gr00t-finetune.md]]
- [[STEP-11-g1-language.md]]
- [[STEP-99-collect-results.md]]
