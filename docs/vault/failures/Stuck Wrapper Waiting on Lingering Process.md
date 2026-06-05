---
tags: [failure, process, wrapper, stuck, isaac-sim]
---

# Stuck Wrapper Waiting on Lingering Process

## Symptom
A shell script doing `while pgrep -af evaluate_seq.py >/dev/null; do sleep 4; done; nohup <next_command> &` never proceeds to the next command. The next job never starts.

## Root Cause
Isaac Sim Python processes linger during shutdown (Omniverse Kit plugin teardown takes several seconds). The `pgrep` on the **host** matches the process name while it's still running in the container OR while Omniverse is doing cleanup. The while-loop never exits.

Also: the nohup command was launched in a sub-shell (`ssh ... '...'`) where the eval was running as a background `nohup` in a previous invocation. The parent shell retained the process reference.

## Fix

1. **Kill the lingering process inside the container** first:
```bash
docker exec isaac-lab-base bash -lc "pkill -f evaluate_seq.py; pkill -f custom_play.py" 2>/dev/null
sleep 3
```

2. **Don't gate critical follow-up work on pgrep polling a sim process.** Instead:
   - After submitting a job with `nohup ... &`, note the PID
   - Poll with `kill -0 $PID` (checks if your specific PID is still alive) rather than pgrep on a name
   - Or just check for the *output file* (the actual result you care about) instead of the process

3. **Launch the next job directly** instead of in a chain:
```bash
# Kill old job
docker exec isaac-lab-base bash -lc "pkill -f evaluate_seq.py" 2>/dev/null
sleep 3
# Start new job directly
docker exec -e PYTHONPATH="$PP" isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/custom_play.py ... &
```

## Lesson
> Isaac Sim processes linger during teardown. Poll for the *output artifact* (file exists?), not the process name.

## Related

- [[Video Render Never Exits]]
- [[Isaac Sim Docker Container]]
- [[00 - Failure Index]]
