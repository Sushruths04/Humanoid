# Results

_Generated 2026-05-30T19:19:49Z_

## Current State

```json
{
  "current_step": "02_gr00t_demo",
  "status": "waiting",
  "phase": 0,
  "started_at": null,
  "last_checkpoint": "2026-05-30T19:19:49Z",
  "notes": "CPU-first thesis execution scaffold. Run thesis/run_thesis.sh --cpu-prep before switching to GPU."
}
```

## Executed Checks

| Step | Status | Evidence |
|---|---|---|
| 00 setup | done | state/00_setup.done |
| 01 GR00T install | done | logs/01_gr00t_import.txt |
| 02 GR00T GPU demo | waiting on Hugging Face auth | results/gr00t_demo/summary.txt |
| 11 G1 language scaffold | done | logs/11_language_import.txt |

## GR00T Demo Summary

```text
# GR00T demo summary

Generated: 2026-05-30T19:19:49Z
Exit status: 1
Blocked by gated Hugging Face access: 1
Log: /home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log
Plot: /home/zeus/content/Humanoid/thesis/results/gr00t_demo/traj_1.jpeg

## Preflight
torch 2.7.1+cu126
cuda_available True
cuda_device NVIDIA L4
gr00t_import ok
policy_import Gr00tPolicy
hf_token_present True

## Key lines
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_demo.log:Cannot access gated repo for url https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json.
/home/zeus/content/Humanoid/thesis/logs/02_gr00t_preflight.log:hf_token_present True
```

## Remaining Blocker

GR00T N1.7 model loading requires Hugging Face authentication and access to gated repo nvidia/Cosmos-Reason2-2B.

```bash
hf auth login
bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave
```

## Step Logs

- [[STEP-00-setup.md]]
- [[STEP-01-gr00t-install.md]]
- [[STEP-02-gr00t-demo.md]]
- [[STEP-11-g1-language.md]]
- [[STEP-99-collect-results.md]]
