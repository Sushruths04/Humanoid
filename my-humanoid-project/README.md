# My Humanoid RL Project

Language-conditioned humanoid loco-manipulation thesis code.

CPU-prep status:
- Deterministic command embeddings are implemented in `my_humanoid_project/language_commands.py`.
- The custom Gymnasium task id is `Humanoid-G1-Language-PickPlace-v0`.
- The task subclasses Isaac Lab's stock `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` when Isaac Lab is available.

GPU handoff:
1. Switch the Lightning Studio to an RTX GPU.
2. Set `USE_GPU=1` and `CPU_PREP_ONLY=0` in `../thesis/config.env`.
3. Run `bash thesis/run_thesis.sh --all`.
