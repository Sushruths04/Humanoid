---
tags: [workflow, reproduce, quickstart, cold-start]
---

# Reproduce From Scratch

Step-by-step recipe to reproduce all results from a completely fresh GPU machine.

---

## Step 1: Connect and verify GPU

```bash
ssh s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# expected: NVIDIA L4, 23034 MiB (or T4, 15360 MiB for nav-only)
```

If SSH fails → [[SSH Key Recovery]].

---

## Step 2: Get the repo

```bash
cd /teamspace/studios/this_studio/Humanoid
git fetch origin && git checkout feat/planned-scripts && git pull
```

---

## Step 3: Pull and start the container

```bash
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag  ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base
cd IsaacLab/docker
touch .isaac-lab-docker-history
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build
cd ../..
docker ps | grep isaac  # should show "Up"
```

If pull fails → [[GHCR Auth Denied]].  
If compose fails → [[container.py Forces Rebuild]].

---

## Step 4: Verify the CPU tests

```bash
cd programs
/home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest -q
# Expected: 34 passed
cd ..
```

---

## Step 5: Smoke test (2-min wiring check)

```bash
SMOKE=1 bash programs/scripts/batch_test_nav.sh
# Runs 16 envs, 2 iterations each — just checks task registration + eval pipeline
```

---

## Step 6: Train all four tasks

```bash
# Either all at once:
bash programs/scripts/batch_test_nav.sh   # ~80 min total

# Or one at a time (recommended, monitor each):
bash programs/scripts/train_eval_nav.sh Humanoid-G1-CommandNav-v0  4096 500 256
bash programs/scripts/train_eval_nav.sh Humanoid-G1-LangNav-v0     4096 500 256
bash programs/scripts/train_eval_nav.sh Humanoid-G1-ObstacleNav-v0 4096 500 256
bash programs/scripts/train_eval_nav.sh Humanoid-G1-SeqNav-v0      4096 500 256
```

Note: SeqNav uses a different evaluator. The batch script calls `evaluate.py` for SeqNav (which crashes). For SeqNav eval, run manually:
```bash
CKPT=$(docker exec isaac-lab-base bash -lc "ls -t $(docker exec ... ls -td .../g1_flat/*/ | head -1)model_499.pt")
docker exec -e PYTHONPATH="..." isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/programs/common/eval/evaluate_seq.py \
  --task Humanoid-G1-SeqNav-v0 --headless --num-envs 256 \
  --checkpoint "$CKPT" --out programs/results/humanoid-g1-seqnav-v0.md
cp programs/results/humanoid-g1-seqnav-v0.md docs/results/
```

---

## Step 7: Upload checkpoints to HF

```bash
hf auth login --token <your_token>
docker cp isaac-lab-base:/workspace/.../model_499.pt _hfstage/g1_commandnav.pt
# ... (copy all 4 checkpoints)
hf upload mitvho09/humanoid-g1-nav _hfstage/g1_commandnav.pt checkpoints/g1_commandnav/model_499.pt
```

---

## Expected Results

| Task | Expected success |
|---|---|
| CommandNav | ~90%+ |
| LangNav | ~95%+ |
| ObstacleNav | ~80%+ |
| SeqNav | ~75%+ full-sequence |

---

## Related

- [[Lightning Studio Environment]]
- [[Isaac Sim Docker Container]]
- [[Training Recipe]]
- [[Evaluation Harness]]
- [docs/PLANNED_SCRIPTS.md](../../docs/PLANNED_SCRIPTS.md)
