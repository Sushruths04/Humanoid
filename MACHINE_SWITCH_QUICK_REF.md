# Machine Switch Quick Reference

Current status:
- Last observed Lightning state: idle
- Last observed training result: `Vision Smoke failed with exit code 255`
- No `custom_train.py` process was running when last checked

## One-Command Flow

On a fresh GPU machine:

```bash
export GITHUB_TOKEN="..."
export HF_TOKEN="..."
cd /home/zeus/content/Humanoid
bash thesis/scripts/machine_switch.sh bootstrap
```

Start training:

```bash
export NUM_ENVS=2048
export MAX_ITERS=5000
bash thesis/scripts/machine_switch.sh train
```

Check status:

```bash
bash thesis/scripts/machine_switch.sh status
```

Sync large outputs to Hugging Face:

```bash
bash thesis/scripts/machine_switch.sh sync
```

Download prior Hugging Face artifacts:

```bash
python3 thesis/scripts/hf_download.py
```

## What Goes Where

- GitHub: code, shell scripts, docs, config
- Hugging Face: checkpoints, logs, rollouts, large artifacts
- Remote GPU machine: execution only

## When To Stop

- Keep the run alive if it is printing iteration metrics and reward/loss values.
- Stop about 30 minutes before the machine window ends.
- Sync to Hugging Face before deleting or switching machines.
