# Compute Guide — Modal vs Lightning AI (+ Docker machine-switching)

## The one principle
> **Lightning = interactive box you SSH into and *debug* on. Modal = fire-and-forget jobs that already work.**
> Always get a task **green on Lightning first** (interactive), then **scale/repeat it on Modal**.

Also remember the two hard constraints:
- **GPU must have RT cores** → **L40S / L4 / A10**. ❌ Never A100/H100 (Isaac Sim won't init/render).
- **Camera/rendering needs Vulkan/graphics** → that path is **Lightning-only** in practice; Modal does the **compute-only** (training) jobs.

## Decision table
| Task | Use | Why |
|---|---|---|
| **First GPU smoke + fixing `# VERIFY` markers** | 🟧 **Lightning** | needs iterative, interactive debugging (edit → rerun in seconds). Modal's function model is wrong for this. |
| Language-on training (2–4 h) | 🟦 **Modal** | short, compute-only, scale-to-zero when done |
| Wakeboard **Stage I** (6–12 h) | 🟦 Modal *or* 🟧 Lightning | both fine; Modal cheaper if you don't babysit (checkpoint+resume) |
| Wakeboard **Stage II** (10–20 h) | 🟧 Lightning *or* 🟦 Modal+resume | long run; Lightning easier to monitor live; Modal ok with the 24h-resume loop |
| **Eval / speed sweep / ablations** | 🟦 **Modal** | many short jobs → fan out in **parallel** (Modal's superpower) |
| **Video / rendered rollout** | 🟧 **Lightning** | needs Vulkan/graphics (the `40_record_video.sh` path) |
| Hyperparameter search | 🟦 **Modal** | parallel functions, pay-per-second |
| Watching the robot in a GUI / WebRTC | 🟧 Lightning | rendering + persistent session |

**Rule of thumb:** *short + parallel + compute-only → Modal. long + interactive + rendering → Lightning.*

## How to actually run

### Modal (main)
```bash
pip install modal && modal token new
modal volume create wakeboard-ckpts
# train Stage I (compute-only, L40S):
modal run modal_app.py --action train --config configs/stage1.yaml
# eval (can fan out many speeds in parallel — call evaluate.remote per speed):
modal run modal_app.py --action eval --checkpoint /ckpts/wakeboard_stage1/model_latest.pt
```
`modal_app.py` already: uses your GHCR image, L40S GPU, a Volume at `/ckpts`, 24h timeout, resume-friendly.

### Lightning AI (debug + long + render)
```bash
# one-time on a fresh machine (reuse your EXISTING image + portability helper):
docker login ghcr.io
bash thesis/scripts/docker_image_portability.sh pull   # pull humanoid-isaaclab (fast switch)
cd wakeboarding-experiment
# debug interactively (fix VERIFY markers):
./docker/run.sh shell
#   (inside) bash scripts/00_smoke.sh
# long training:
./docker/run.sh train stage2
# render a video (needs graphics — Lightning only):
CKPT=checkpoints/.../model_latest.pt bash scripts/40_record_video.sh
```

## Docker = fast machine switching (reuse the ONE existing image)
The wakeboarding task is the **same system** (Isaac Lab + G1 + RSL-RL) as the rest of the
Humanoid repo, so it reuses the **existing** image `ghcr.io/sushruths04/humanoid-isaaclab:latest`
— no second image. Image build + push/pull/save/load is already handled by your existing
helper; do not duplicate it:

```bash
# push from the machine where it works:
bash thesis/scripts/docker_image_portability.sh push
# pull on a new machine (the switch — one command):
bash thesis/scripts/docker_image_portability.sh pull
# tarball fallback if no registry:
bash thesis/scripts/docker_image_portability.sh save   # then ... load
```
Code is **mounted** at runtime (docker-compose), so editing code never needs a rebuild.
Modal uses the **same** image automatically (`modal.Image.from_registry(...)` in
`modal_app.py`), so Modal and Lightning never drift apart. The only thing the wakeboarding
task may add on top is `rsl-rl` — and only if it's missing from the base (see
`docker/Dockerfile`, which is OPTIONAL).

## Cost intuition (not exact prices)
- **Modal** bills per-second and scales to zero → cheapest for **bursty/short/parallel** work and for jobs you don't watch.
- **Lightning** is a persistent rented box → better for **long interactive** sessions and **rendering**, and you're not paying cold-start/idle churn while actively debugging.
- For a >~50%-utilization long run, a persistent box is often cheaper than serverless; for sporadic eval/sweeps, serverless wins.

## Suggested end-to-end flow for THIS project
1. **Lightning**: `00_smoke.sh` interactively, fix `# VERIFY` markers until green. *(½–1 day, one-time)*
2. **Push the fixed image** (`run.sh push`) so Modal has the working version.
3. **Modal**: run Stage I with the pull-speed curriculum (checkpoint to Volume).
4. **Modal**: run Stage II (resume-friendly) — or Lightning if you want to watch it.
5. **Modal**: fan out the eval speed-sweep + ablations in parallel.
6. **Lightning**: record the demo video (graphics path).
7. Update `vault/03_Results/Results_Live.md` per `DOC_PROTOCOL.md`.
