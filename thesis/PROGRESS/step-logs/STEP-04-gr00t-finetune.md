# STEP 04 GR00T finetune command gate

_2026-05-30T19:33:51Z_

## GPU run template

Verify the current upstream fine-tune command before execution.

```bash
cd /home/zeus/content/Humanoid/Isaac-GR00T
nvidia-smi
# Example shape, adjust to current README:
python scripts/gr00t_finetune.py --output-dir /home/zeus/content/Humanoid/thesis/checkpoints/gr00t --max-steps 2000 --save-interval 500
```

Exit intentionally until the current README command is verified.

