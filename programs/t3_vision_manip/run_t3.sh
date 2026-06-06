#!/bin/bash
# T3 Vision Manipulation — one-shot pipeline
# Usage: MUJOCO_GL=egl PYTHONUNBUFFERED=1 bash programs/t3_vision_manip/run_t3.sh > t3_run.log 2>&1
set -e

REPO=/teamspace/studios/this_studio/Humanoid
PY=/home/zeus/miniconda3/envs/groot_env/bin/python
DATA=/teamspace/studios/this_studio/libero_datasets/libero_spatial
CKPT=$REPO/programs/checkpoints/t3_pixel_bc/pixel_bc.pt
RESULT_DOC=$REPO/docs/results/t3_pixel_bc.md

cd $REPO

echo "=== [T3] Step 1/2: Train pixel BC policy ==="
$PY -u -m programs.t3_vision_manip.train_pixel_bc \
    --data-dir $DATA \
    --out $CKPT \
    --epochs 200 \
    --batch-size 64 \
    --result-doc $RESULT_DOC

echo "=== [T3] Step 2/2: Evaluate pixel BC policy ==="
MUJOCO_GL=egl $PY -u -m programs.t3_vision_manip.evaluate_pixel_bc \
    --checkpoint $CKPT \
    --task libero_spatial \
    --num-envs 10 \
    --out $RESULT_DOC \
    --video-dir $REPO/programs/videos/t3_pixel_bc

echo "=== [T3] Committing results ==="
git add programs/checkpoints/t3_pixel_bc/ docs/results/t3_pixel_bc.md \
        programs/videos/t3_pixel_bc/ 2>/dev/null || true
git commit -m "T3: pixel BC policy on LIBERO Spatial — $(grep 'Mean task success' $RESULT_DOC | head -1)"
git push origin feat/planned-scripts

echo "=== [T3] COMPLETE ==="
