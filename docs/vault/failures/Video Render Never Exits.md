---
tags: [failure, video, render, play, process]
---

# Video Render Never Exits

## Symptom
`custom_play.py --video` runs, the mp4 file appears in `logs/.../videos/play/rl-video-step-0.mp4`, but the process keeps running indefinitely. The script never exits on its own.

## Root Cause
`play.py` (the stock Isaac Lab RSL-RL play script) contains a `while simulation_app.is_running(): ...` loop. After the video is saved (after `video_length` steps), the loop continues running the policy forever — there's no automatic exit after recording.

## Fix
Watch for the mp4 file to appear, then kill the process manually:

```bash
# Wait and watch for the file
docker exec isaac-lab-base bash -lc \
  "find /workspace/isaaclab/logs -name '*.mp4' 2>/dev/null -exec ls -la {} \;"
# Once it appears, kill the process
docker exec isaac-lab-base bash -lc "pkill -f custom_play.py"
# Copy out the file
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/videos/play/rl-video-step-0.mp4 \
  docs/results/videos/demo.mp4
```

## Additional Gotcha: stdout not flushed before app.close()
`simulation_app.close()` hard-exits the process. Any `print()` statements after that are never flushed to stdout. If you add diagnostic prints to custom_play.py, **write them to a file with explicit flush**:
```python
with open("/workspace/programs/results/_diag.txt", "w") as f:
    f.write("result: %s\n" % value)
    f.flush()
import sys; sys.stdout.flush()
env.close(); app.close()
```

## Related
- [[Rendering Demo Videos]]
- [[Stuck Wrapper Waiting on Lingering Process]]
- [[Isaac Sim Docker Container]]
