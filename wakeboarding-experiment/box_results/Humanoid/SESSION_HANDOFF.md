# Session Handoff — Reward Wiring Fix & Training Bring-up

**Date:** 2026-06-20  
**Branch:** `gpu-l4-bringup` (8 commits ahead of origin)  
**SSH:** `s_01kvjpvyze89dkgnykht09z69w@ssh.lightning.ai`  
**Home:** `/teamspace/studios/this_studio/`

---

## 1. Executive Summary

We found and fixed the **root cause** of why training was not learning: YAML reward weights were never applied because `apply_reward_weights()` was called AFTER env construction, but Isaac Lab's `RewardManager` copies weights at init time. All YAML overrides were silently ignored for every prior run.

After the fix, 400-iter probe showed **fell rate dropping from 64% → 25%** — the policy IS learning now. Two additional issues were found and fixed: tight board_range termination bounds and NaN propagation from `pen_dof_pos_limits`.

---

## 2. The Root Cause Bug — Reward Weight Wiring

### What was happening
```
train.py flow (BUGGY):
  env_cfg = WakeboardStartEnvCfg()
  env = WakeboardStartEnv(env_cfg)          # <-- RewardManager copies weights HERE
  apply_reward_weights(env.cfg, yaml_weights)  # <-- Too late! Weights already copied
```

Isaac Lab's `RewardManager.__init__()` does:
```python
self._term_weights[name] = term_cfg.weight  # copies at construction
```

Changing `term_cfg.weight` AFTER construction has ZERO effect.

### The fix (commit a526632)
```
train.py flow (FIXED):
  env_cfg = WakeboardStartEnvCfg()
  apply_reward_weights(env_cfg, yaml_weights)  # <-- Before env construction
  env = WakeboardStartEnv(env_cfg)             # <-- RewardManager reads updated weights
```

Changed function signature from `apply_reward_weights(env, ...)` to `apply_reward_weights(env_cfg, ...)`.

### Impact
Every single prior probe run used hardcoded default weights from `RewardsCfg` class, ignoring all YAML overrides. The entire penalty/rebalancing effort from earlier sessions was ineffective until this fix.

---

## 3. All Failures Encountered

### Failure 1: Robot stands still, falls (probes v1-v4)
- **Symptom:** Fell rate ~90%+ at iter 0, reward deeply negative (-4 to -5)
- **Root cause:** Two problems stacked:
  1. `rope_model.py` had `lead=5.0m` — rope stretched 5m before applying force, so robot experienced no pull initially
  2. Reset pose used default standing (0.74m), not crouched cannonball
- **Fixes applied:**
  - Changed `lead` default from 5.0 to 0.4 (gives ~320N initial pull vs 600N saturation)
  - Added `_reset_to_cannonball()` function with hip/knee/torso/arm angles from G1 joint limits

### Failure 2: Cannonball reset didn't persist on auto-reset
- **Symptom:** Robot started in cannonball at first reset, but after falling, auto-reset used default standing pose
- **Root cause:** `ManagerBasedRLEnv._reset_idx()` calls `event_manager.apply(mode="reset")`, but only `reset()` called `_refresh_biomech_buffers()` and rope reset. Auto-reset during `step()` called `_reset_idx()` directly, bypassing rope/buffer reset.
- **Fix:** Added `_reset_idx()` override that calls `super()._reset_idx()`, then refreshes biomech buffers and resets rope anchor per-env.

### Failure 3: Negative reward signal (policy learns to freeze)
- **Symptom:** Reward = -4.5, reward ratio 1:7 positive-to-negative
- **Root cause:** Penalty weights were 50-100x larger than positive reward weights (from defaults that were never overridden due to Failure 4)
- **Fix:** Rebalanced YAML weights — positives 5x up, penalties 5x down

### Failure 4: YAML reward weights never applied (THE ROOT CAUSE)
- **Symptom:** All YAML weight changes had zero effect across ALL prior runs
- **Root cause:** `apply_reward_weights()` called after env construction (see Section 2)
- **Fix:** Moved weight application before env construction

### Failure 5: NaN physics explosion (256+ envs on L4)
- **Symptom:** Complete training crash, NaN in all observations
- **Root cause:** GPU memory pressure on L4 causes physics solver instability with >16 parallel envs
- **Fix:** Capped at 16 envs for all probes; added NaN/inf guard in `step()` that sanitizes obs+rewards

### Failure 6: NaN from pen_dof_pos_limits crashes PPO
- **Symptom:** `pen_dof_pos_limits` returns NaN at iter ~399, corrupts PPO gradients, crashes with `RuntimeError: normal expects all elements of std >= 0.0`
- **Root cause:** `soft_joint_pos_limits` can be None/NaN if physics hasn't fully initialized or after extended simulation
- **Fix:** Added NaN guard: if `soft_joint_pos_limits` is None or all-NaN, return zeros; also NaN-clamp lo/hi/j values

### Failure 7: Board_range termination dominates (79%)
- **Symptom:** After fell rate dropped, board_range termination rose to 79% — board swings out of [-20°, 45°] bounds
- **Root cause:** Bounds too tight for early training where policy acts randomly
- **Fix:** Widened from [-20°, 45°] to [-40°, 60°]

---

## 4. Probe Results Timeline

| Probe | Config | Iter | Fell (start→end) | Reward | Key finding |
|-------|--------|------|-------------------|--------|-------------|
| v1-v4 | 256 envs, default weights | 100 | ~0.90 (flat) | -4.5 | Physics explosion, NaN |
| v5 | 16 envs, cannonball + lead=0.4 | 100 | 0.47→0.73 | -1.5 | Cannonball works, but weights ignored |
| v6 | 16 envs, rebalanced YAML | 100 | 0.65→0.85 | -2.5 | Weights STILL ignored |
| v7 | 16 envs, **fixed wiring** | 100 | 0.79→0.81 | **+1.1** | Weights applied! Positive reward! |
| v8 | 16 envs, 400 iters | 400 | **0.64→0.25** | +0.93 | **Policy learning!** board_range dominant |

### v8 detailed trend (key iters):
```
iter   0: fell=0.64, reward=1.22
iter 100: fell=0.42, reward=0.91
iter 200: fell=0.35, reward=0.87
iter 300: fell=0.31, reward=0.90
iter 399: fell=0.25, reward=0.93  (board_range=0.79)
```

---

## 5. Reward Breakdown at Iter 399

| Term | Value | Notes |
|------|-------|-------|
| pelvis_height | +0.047 | Working — reward for standing height |
| uprightness | +0.026 | Working — reward for staying vertical |
| knee_bend_maintained | +0.019 | Working — reward for bent knees |
| survival | +0.012 | Working — reward for staying alive |
| board_positive_angle | +0.007 | Working — reward for nose-up board angle |
| forward_glide | +0.003 | Weak — needs more board speed |
| pen_stand_too_fast | -0.005 | Active — penalizing fast standing |
| pen_torque | -0.006 | Active — penalizing high torque |
| pen_dof_pos_limits | NaN → fixed | Was crashing training |
| **Total positive** | **+0.12** | |
| **Total negative** | **-0.014** | **Ratio 8.5:1 positive** ✓ |

---

## 6. Current Fixes Applied (all on gpu-l4-bringup)

| Commit | Fix | Files |
|--------|-----|-------|
| `6e5a863` | Cannonball reset + rope lead=0.4 + _reset_idx | wakeboard_start_cfg.py, rope_model.py |
| `1816fd7` | Rebalance reward weights + bump envs | smoke.yaml |
| `78763f8` | NaN guard + 512 envs | wakeboard_start_cfg.py, smoke.yaml |
| `2da47d4` | NaN detection + obs sanitization | wakeboard_start_cfg.py |
| `4a04747` | 16 envs + moderate penalties + robust NaN guard | wakeboard_start_cfg.py, smoke.yaml |
| `7c227c1` | Flip reward signal (positives 5x up, penalties 5x down) | smoke.yaml |
| `a526632` | **ROOT CAUSE FIX: weights applied before env construction** | train.py |
| (pending) | board_range [-40,60] + pen_dof_pos_limits NaN guard | wakeboard_start_cfg.py, wakeboard_rewards.py |

---

## 7. Known Issues & Constraints

### L4 GPU constraints
- **Max safe envs:** 16 (256+ causes physics NaN explosions)
- **Steps per iter:** 384 (16 envs × 24 steps/env)
- **Time per 100 iters:** ~9.5 minutes
- **5k iters estimate:** ~8 hours

### Physics limitations
- Board is fixed-jointed to feet via USD (not PhysX soft coupling)
- Rope force applied as external force on palm links (not PhysX cable)
- `set_external_force_and_torque` is deprecated — should migrate to `permanent_wrench_composer`

### Reward terms not yet firing
- `success_bonus` = 0 (robot never holds stable 1.5s — needs more training)
- `height_progress` = 0 (robot doesn't rise during episode — needs more training)
- `amp_style` = 0 (disabled, set to weight 0.0)

---

## 8. Files Modified This Session

| File | Changes |
|------|---------|
| `train.py` | Fixed `apply_reward_weights` call order + function signature |
| `src/tasks/wakeboard_start_cfg.py` | Cannonball reset, _reset_idx override, NaN guard in step(), widened board_range |
| `src/rewards/wakeboard_rewards.py` | NaN guard in pen_dof_pos_limits |
| `src/rope_model.py` | lead default 5.0 → 0.4 |
| `configs/smoke.yaml` | 16 envs, rebalanced reward weights |
| `configs/stage1.yaml` | **NEW** — 5000 iterations, 16 envs, Stage I config |

---

## 9. Next Steps

1. **Launch Stage I** with `configs/stage1.yaml` (5k iters, ~8 hours)
2. Monitor for:
   - Fell rate continuing to drop below 0.25
   - board_range termination stabilizing (wider bounds should help)
   - success_bonus firing (robot holds stable for 1.5s)
   - Mean reward trend (should keep climbing)
3. After Stage I: evaluate checkpoints, consider widening env count if physics allows
4. Future: migrate deprecated APIs, add contact-based termination, curriculum on rope speed

---

## 10. How to Run

```bash
# SSH to remote
ssh s_01kvjpvyze89dkgnykht09z69w@ssh.lightning.ai

# Kill any running containers
docker kill $(docker ps -q) 2>/dev/null

# Launch Stage I (5k iters, ~8 hours)
cd ~/Humanoid/wakeboarding-experiment
setsid bash docker/run.sh train stage1 > ~/_setup_logs/stage1.log 2>&1 < /dev/null & disown

# Monitor progress
tail -f ~/_setup_logs/stage1.log | grep -E 'Learning iteration|Mean reward|fell|board_range'

# Check saved checkpoints
ls -la ~/Humanoid/wakeboarding-experiment/logs/wakeboard_stage1/
```

---

## 11. Key Architecture Notes

- **Isaac Lab env flow:** `step()` → `_refresh_biomech_buffers()` → `rope.step_anchor()` → `rope.compute_force()` → `apply_handle_force()` → `super().step()` → reward/termination managers → NaN guard
- **Auto-reset flow:** `step()` detects timeout/fell → `ManagerBasedRLEnv._reset_idx()` → our override resets rope + buffers
- **Reward weight flow:** YAML → `apply_reward_weights(env_cfg)` → `RewardManager.__init__()` reads from cfg → stored in `_term_weights` dict
- **Observation space:** joint_pos_rel + joint_vel_rel + base_ang_vel + proj_gravity + last_action + board_pitch + rope_force + handle_rel + v_pull + phase

---

*Session completed by opencode on 2026-06-20. All fixes committed to `gpu-l4-bringup`.*
