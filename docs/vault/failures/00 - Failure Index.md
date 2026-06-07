---
tags: [failures, index, lessons]
---

# Failure Index

Quick-reference table. Click the link for the full story.

| # | Failure | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | [[SeqNav Stand-Still Local Optimum]] ⭐ | full-seq 0.4%, robot barely moves | training bootstrap failure; targets too far to reach before timeout | closer targets (1–2.5m) |
| 2 | [[Decorative Navigation Defect]] | robot "navigates" but ignores command | reward tracked velocity, not target | rebuilt with genuine command conditioning |
| 3 | [[Results Lost to Ephemeral Container Storage]] | exit 0, no result file on host | `docs/` not bind-mounted; container storage ephemeral | write to `programs/results/` (bind-mounted) |
| 4 | [[Eval Crash - Missing Buffer]] | `AttributeError: _nav_target_ids` on SeqNav eval | evaluate.py assumes CommandNav buffers | wrote evaluate_seq.py for sequential tasks |
| 5 | [[GHCR Auth Denied]] | `docker pull` → `denied` | stored token revoked (was pasted into chat) | made package public |
| 6 | [[container.py Forces Rebuild]] | build fails needing NGC base image | `container.py start` always runs `--build` | bypass with raw `docker compose --no-build` |
| 7 | [[SSH Heredoc Apostrophe Corruption]] | file broken mid-write, SyntaxError | apostrophe in content closed single-quoted SSH arg | scp files instead of SSH heredocs |
| 8 | [[Video Render Never Exits]] | `custom_play.py` runs forever after mp4 saved | `play.py` loop is infinite | grab mp4, then kill the process |
| 9 | [[Stuck Wrapper Waiting on Lingering Process]] | follow-up job never starts | while-pgrep loop blocked by lingering sim process | kill in-container python; don't gate on pgrep |
| 10 | [[SSH Key Drops on Restart]] | `Permission denied (publickey)` | Studio restart drops key auth | re-run PowerShell ssh-gen/ssh-public URLs |

---

## P3 VisionNav Failures (added 2026-06-07)

| # | Failure | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 11 | [[RTX BVH Hang at High Env Count]] ⭐ | training hangs 30+ min at startup | RTX BVH build scales super-linearly with env count | `--num_envs 4096` max |
| 12 | [[RTX Rendering is the Bottleneck (Not CUDA Cores)]] ⭐ | A100 at 14% GPU util, 66s/iter | RT cores saturated by 67M-pixel tiled render | 128→64 resolution (5.1× speedup) |
| 13 | [[OOM With Camera Rollout Buffer]] | OOM during PPO update at 78.4 GB | 4096×48×128²×3 image buffer exhausts VRAM | `num_mini_batches=64` or use 64×64 |
| 14 | [[update_period Does Not Reduce Render Time]] | no speedup from 5 Hz camera setting | `update_period` controls Python reads, not RTX renders | reduce resolution instead |
| 15 | [[Docker Image Lost on GPU Upgrade]] | `No such image` after switching machine type | Docker images stored on ephemeral VM disk, not persistent storage | re-pull on every new machine |
| 16 | [[RSL-RL Resume Resets Loop Counter]] | ETA 1:49 not 37 min after `--resume` | `max_iterations` = new iters to run, not target iteration | accept extra iters — warm start converges faster |
| 17 | [[play.py Fails - Custom Task Not Registered]] | `NameNotFound: Humanoid-G1-VisionNav` | stock play.py never imports custom task module | use `custom_play.py` as entry point |
| 18 | [[Python File Corruption Over SSH - Use Python Write]] | SyntaxError after SSH heredoc write | shell quoting layers mangle escape sequences | write via `python3 -c "open(...).write(...)"` or scp |
| 19 | [[HuggingFace CLI Deprecated]] | `command not found: huggingface-cli` | CLI deprecated in newer huggingface_hub | use `HfApi().upload_folder(...)` in Python |
| 20 | [[Git LFS Mismatch on New Studio]] | 200+ modified files on fresh studio | studio has real binaries but git expects LFS pointers | `GIT_LFS_SKIP_SMUDGE=1 git reset --hard origin/branch` |
| 21 | [[play.py Checkpoint Bare Filename Not Found]] | `FileNotFoundError: model_499.pt` on eval | `--checkpoint` triggers `retrieve_file_path()` which needs full path | use `--load_run` only; or pass absolute path |
| 22 | [[play.py Eval Has No Episode Output]] | eval runs forever, 33% GPU, no stats printed | play.py episode loop has no print statements; buffered stdout never flushes | use training stats (more robust); or write custom N-episode eval script |

---

## Lessons Distilled

1. **Watch the per-term reward breakdown**, not just total reward. Flat task term + rising total = reward farming.
2. **Make the first success easy** in multi-step tasks — curriculum, closer targets, whatever it takes.
3. **Bind mounts only** — never write important outputs to unmounted container paths.
4. **Never paste tokens into chat** — rotate immediately if you do, then make the resource public.
5. **scp for code transfer**, never SSH heredocs with complex content.
6. **Isaac Sim processes linger** — kill in-container, don't poll host pgrep.
7. **Behavior probe** > test accuracy — verify the policy CHANGES BEHAVIOR with the command.
8. **High VRAM ≠ high GPU utilization** — profile RT cores vs CUDA cores separately for camera-based RL.
9. **Reduce camera resolution before scaling envs** — resolution has 4× the impact on RTX render time.
10. **Re-pull Docker image on every new machine** — Lightning VM switches wipe Docker cache completely.
11. **`max_iterations` in RSL-RL = new iters to run**, not the target iteration number.
12. **Always use `custom_train.py` / `custom_play.py`** — never call Isaac Lab stock scripts directly for custom tasks.
13. **Kill training process before eval** — Isaac Sim holds VRAM until the process exits; lingering train process causes OOM on eval even after training completes.
14. **`--checkpoint` in play.py needs a full path** — bare filename triggers `retrieve_file_path()` not `get_checkpoint_path()`; use `--load_run` only for cleanest eval invocation.
15. **play.py episode loop has no output** — training stats (`time_out` fraction) are the authoritative success metric; write a custom N-episode eval script if a separate eval run is needed.

---

## Related

- [[00 - START HERE]]
- [[Common Failure Patterns]]
