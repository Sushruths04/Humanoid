---
tags: [failure, docker, storage, ephemeral, data-loss]
---

# Results Lost to Ephemeral Container Storage

## Symptom

`train_eval_nav.sh` exits 0 (success). But `docs/results/` on the host is empty. The result file doesn't exist.

## Root Cause

The eval script was called with `--out docs/results/humanoid-g1-obstaclenav-v0.md`. Inside the container, this relative path resolves to `/workspace/isaaclab/docs/results/...` (the container's working directory is `/workspace/isaaclab`). That path is **not bind-mounted** to the host.

When the container is removed (or the machine is restarted), this file is gone.

The bind-mounted paths are only:
- `/workspace/programs` ↔ host `programs/`
- `/workspace/my-humanoid-project` ↔ host `my-humanoid-project/`

`docs/` is NOT in that list.

## The Fix

Write results to the **bind-mounted** path instead:

```bash
# In train_eval_nav.sh:
OUT_C="/workspace/programs/results/${NAME}.md"   # bind-mounted → persists on host
docker exec ... evaluate.py ... --out "$OUT_C"
# then mirror to docs/results/ on the host:
cp "programs/results/${NAME}.md" "docs/results/${NAME}.md"
```

Also:
- Add `programs/results/` to `.gitignore` (container writes are root-owned; only the host-owned mirror in `docs/results/` gets committed)
- Run `docker exec ... chown -R $(id -u):$(id -g) /workspace/programs/results` after container-side writes to restore host ownership

## Verification Test

```bash
# Write a file from inside container, check it appears on host
docker exec isaac-lab-base bash -lc "echo test > /workspace/programs/results/_test.txt"
cat programs/results/_test.txt   # should print "test"
```

## Lesson

> **Know exactly which paths are bind-mounted.** Before any important output, verify it will land in a mounted path. Relative paths in a container are a trap.

Container ephemeral storage looks and feels like a normal filesystem. There's no warning when it disappears. Always write important artifacts to bind-mounted paths or copy them out immediately.

## Related

- [[Isaac Sim Docker Container]]
- [[Lightning Studio Environment]]
- [[00 - Failure Index]]
