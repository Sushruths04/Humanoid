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
# one-time on a fresh machine:
docker login ghcr.io
./docker/run.sh pull                 # pull the prebuilt image (fast machine switch)
# debug interactively (fix VERIFY markers):
./docker/run.sh shell
#   (inside) bash scripts/00_smoke.sh
# long training:
./docker/run.sh train stage2
# render a video (needs graphics — Lightning only):
CKPT=checkpoints/.../model_latest.pt bash scripts/40_record_video.sh
```

## Docker = fast machine switching (build once, run anywhere)
The whole point: **one image** (`ghcr.io/sushruths04/wakeboard-isaaclab:latest`) used by **both** Modal and Lightning. Deps are baked in (slow part, cached); **code is mounted at runtime**, so editing code never triggers a rebuild.

**Set it up once:**
```bash
./docker/run.sh build      # layer rsl-rl + deps on top of your humanoid-isaaclab base
docker login ghcr.io
./docker/run.sh push       # -> GHCR
```
**On any new Lightning machine (the switch):**
```bash
docker login ghcr.io
./docker/run.sh pull       # 1 command, you're ready
./docker/run.sh shell
```
**No registry? tarball fallback:** `./docker/run.sh save` → copy `wakeboard-image.tar` → `./docker/run.sh load`.

Modal uses the **same** image automatically (`modal.Image.from_registry(...)` in `modal_app.py`), so Modal and Lightning never drift apart.

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
