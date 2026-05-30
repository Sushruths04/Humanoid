# Results

_Last Updated: 2026-05-30T22:45:00Z_

## Current State

```json
{
  "current_step": "99_collect_results",
  "status": "done",
  "phase": 2,
  "started_at": "2026-05-30T22:02:00Z",
  "last_checkpoint": "2026-05-30T22:39:12Z",
  "notes": "Full-scale fine-tuning (10,000 steps) and Isaac Lab G1 VLA pipeline completed."
}
```

## Executed Checks

| Step | Status | Evidence |
|---|---|---|
| 00 setup | done | state/00_setup.done |
| 01 GR00T install | done | logs/01_gr00t_import.txt |
| 02 GR00T GPU demo | done | results/gr00t_demo/summary.txt |
| 03 GR00T data seed | done | data/gr00t_language_seed/instructions.jsonl |
| 04 GR00T fine-tune | done | [Hugging Face](https://huggingface.co/mitvho09/GR00T-Humanoid) / `checkpoint-10000` |
| 05 GR00T evaluate | done | results/gr00t_eval_smoke/summary.txt |
| 10 G1 baseline | done | checkpoints/g1_baseline/model_latest.pt |
| 11 G1 language scaffold | done | logs/11_language_import.txt |
| 12 G1 language train | done | checkpoints/g1_language/model_latest.pt |
| 20 G1 custom task | done | checkpoints/g1_custom/model_latest.pt |
| 99 collect results | done | FINAL_RESULTS.md |

## Final Summary

Phase 1 (GR00T replication) and Phase 2 (Isaac Lab G1 VLA-style conditioning) are fully complete.

### Achievements
1.  **Full-Scale Foundation Model**: Successfully fine-tuned the 2B humanoid foundation model for 10,000 steps (Loss: 0.0855).
2.  **Hugging Face Integration**: Model weights (8.8GB) are securely backed up on HF Hub.
3.  **Infrastructure**: Established a robust Isaac Lab Docker environment on Lightning AI with L40S GPU.
4.  **VLA Pipeline**: Implemented a scalable language-commanded observation system for G1.
5.  **Custom Task**: Demonstrated extensibility by implementing a marker navigation task gated by language embeddings.


## Step Logs

- [[STEP-00-setup.md]]
- [[STEP-01-gr00t-install.md]]
- [[STEP-02-gr00t-demo.md]]
- [[STEP-03-gr00t-gendata.md]]
- [[STEP-04-gr00t-finetune.md]]
- [[STEP-05-gr00t-eval.md]]
- [[STEP-10-g1-baseline.md]]
- [[STEP-11-g1-language.md]]
- [[STEP-12-g1-train-eval.md]]
- [[STEP-20-custom-task.md]]
- [[STEP-99-collect-results.md]]
