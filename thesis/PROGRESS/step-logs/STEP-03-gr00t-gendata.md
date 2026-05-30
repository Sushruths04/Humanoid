# STEP 03 GR00T data seed

_2026-05-30T19:33:51Z_

## CPU seed manifest

Created a small language-instruction manifest. This is not a full LeRobot dataset yet; it is the CPU-side seed used by the GPU/data-import step.

```text
200 /home/zeus/content/Humanoid/thesis/data/gr00t_language_seed/instructions.jsonl
{"episode_id": 0, "language_instruction": "pick up the red cube", "source": "cpu_seed_manifest", "status": "needs_lerobot_observations_on_gpu_or_dataset_import"}
{"episode_id": 1, "language_instruction": "pick up the blue cube", "source": "cpu_seed_manifest", "status": "needs_lerobot_observations_on_gpu_or_dataset_import"}
{"episode_id": 2, "language_instruction": "walk to the cube", "source": "cpu_seed_manifest", "status": "needs_lerobot_observations_on_gpu_or_dataset_import"}
{"episode_id": 3, "language_instruction": "stand still", "source": "cpu_seed_manifest", "status": "needs_lerobot_observations_on_gpu_or_dataset_import"}
{"episode_id": 4, "language_instruction": "pick up the red cube", "source": "cpu_seed_manifest", "status": "needs_lerobot_observations_on_gpu_or_dataset_import"}
```

- [x] Language labels exist
- [ ] Convert/import observations/actions into LeRobot format

