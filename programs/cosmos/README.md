# P4 — NVIDIA Cosmos world-model integration (scaffold)

Status: scaffold. Requires a Cosmos environment (separate from Isaac Lab) and
the `nvidia-cosmos/cosmos-predict2.5` cookbook. Heaviest phase; prefer an
A100-80GB burst for the LoRA post-train (see `docs/GPU_VRAM_REQUIREMENTS.md`).

## Pipeline (maps to the cosmos-cookbook)
1. `export_data.py` — export G1 (frames, actions) rollouts into the cookbooks
   action-conditioned format.
2. Inference baseline — run stock Cosmos-Predict 2.5 (2B) to generate a video.
3. `post_train.py` — LoRA post-train (Robot/Policy recipe) on the exported data.
4. Synthetic-trajectory augmentation — generate trajectories, fold into nav/manip
   training, measure the policy gain (headline result).
5. Planning/eval — short-horizon CEM/MPC over the learned model; predicted-vs-real
   eval correlation.

## References
- https://github.com/nvidia-cosmos/cosmos-predict2.5
- https://github.com/nvidia-cosmos  (cosmos-cookbook)

The two skeleton scripts below define the intended CLI; fill in against the
installed cookbook APIs (do not run without the Cosmos environment).
