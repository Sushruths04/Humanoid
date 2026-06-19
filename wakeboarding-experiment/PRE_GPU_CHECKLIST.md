# Day-1 GPU Checklist (run this top-to-bottom when the GPU is up)

> Goal: turn your first paid GPU session into a checklist, not exploration. Pure-PyTorch
> logic is already CPU-verified; this list is only the GPU-runtime glue. Do it on an
> **interactive Lightning box** (not Modal — you're debugging). Budget ~2–4 h.

## 0. Get the box ready (5 min)
```bash
docker login ghcr.io
bash thesis/scripts/docker_image_portability.sh pull   # reuse the existing humanoid-isaaclab image
cd wakeboarding-experiment
docker compose -f docker/docker-compose.yaml run --rm wakeboard bash   # interactive shell
python -c "import rsl_rl, isaaclab; print('deps ok')"   # if rsl_rl missing: pip install rsl-rl-lib
```

## 1. Run the smoke test (it WILL fail first — that's expected)
```bash
bash scripts/00_smoke.sh    # 16 envs, 2 iters
```
Then fix the runtime-only `# VERIFY` markers, in this likely-failure order:

| Order | File / marker | What to confirm on GPU | How |
|---|---|---|---|
| 1 | `wakeboard_start_cfg.py` G1 names | elbow/knee/foot/hand/torso names match `g1.usd` | print `robot.joint_names` + `robot.body_names` once; fix `G1_*` constants if needed |
| 2 | `board.py` foot binding | board welds to `*_ankle_roll_link` and is stable | inspect sim; tune solver iters if jittery |
| 3 | `wakeboard_start_cfg.py::_apply_handle_force` | `set_external_force_and_torque` signature + `_hand_body_ids` | confirm force shows up at the hands |
| 4 | `_quat_pitch` quat order | isaaclab quats are wxyz | sanity-check board pitch sign |
| 5 | obs `loco_mdp.*` names | `joint_pos_rel`, `projected_gravity`, etc. exist in installed version | adjust imports if renamed |
| 6 | reset event | replace `reset_scene_to_default` with the **cannonball crouch** init pose | author the crouch joint targets |

✅ **Smoke is GREEN when** `model_*.pt` is written under `runs/wakeboard_smoke/` with no exception.

## 2. Push the working image (so Modal matches)
```bash
# from repo root, on the box where smoke passed:
bash thesis/scripts/docker_image_portability.sh push
```

## 3. Stage I training (Modal or here)
```bash
bash scripts/10_train_stage1.sh     # 4096 envs, curriculum 10->30 km/h
# or on Modal:  modal run modal_app.py --action train --config configs/stage1.yaml
```
Watch: success rate climbing as the curriculum advances. Save `ckpt_20_stage1_30`.

## 4. Reference motion + AMP, then Stage II
- Author 3 crouch→mid→tall keyframes in `src/amp/reference_motion.py::build_keyframe_reference`
  using the now-known G1 joint layout.
- `STAGE1_CKPT=... bash scripts/20_train_stage2.sh`

## 5. Eval + video + docs
```bash
CKPT=... bash scripts/31_eval_speed_sweep.sh    # Table A (Modal: fan out in parallel)
CKPT=... bash scripts/40_record_video.sh        # LIGHTNING ONLY (needs Vulkan)
bash scripts/99_collect_results.sh              # -> paste into vault/03_Results/Results_Live.md
```

## Remaining `# VERIFY` markers (all runtime-only — can't be resolved without GPU)
- `set_external_force_and_torque` body-id wiring (`_apply_handle_force`)
- foot→board fixed-joint creation (PhysX stability)
- cannonball reset-pose event
- `loco_mdp` term names for the installed Isaac Lab version
- gripper/hand link name (`G1_HAND_LINKS`) if not `*_rubber_hand`

Everything else (rope physics, curriculum, AMP discriminator, reward math, G1 joint-name
resolution) is already done and CPU-tested.
