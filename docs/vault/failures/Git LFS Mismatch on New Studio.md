---
tags: [failure, git, lfs, lightning-studio, p3]
---

# Git LFS Mismatch on New Studio

## Symptom
On a freshly connected Lightning Studio, `git status` showed **200+ modified files** — all binary assets (`.jpg`, `.png`, `.mp4`, `.npz`). No actual changes had been made:
```
 M IsaacLab/docs/source/_static/NVIDIA-logo-black.png
 M IsaacLab/docs/source/_static/benchmarks/cartpole.jpg
 M programs/videos/demo_reel.mp4
... (200+ more)
```
Running `git pull` then failed with `fatal: Need to specify how to reconcile divergent branches`.

## Root Cause
Git LFS tracks large binary files as small pointer text files (e.g., `oid sha256:abc123...`). When a studio doesn't have `git-lfs` properly configured, or the LFS objects were downloaded (replacing pointer files with real binaries), git sees the actual binary content where it expects a small text pointer — hence `M` (modified).

The "divergent branches" error meant this studio also had local commits from a previous session that weren't on the remote.

## Fix: Hard Reset with LFS Skip
```bash
cd /teamspace/studios/this_studio/Humanoid

# Skip LFS smudge filter (don't try to download actual binary content)
GIT_LFS_SKIP_SMUDGE=1 git fetch origin

# Discard all local changes and local-only commits — match remote exactly
GIT_LFS_SKIP_SMUDGE=1 git reset --hard origin/feat/planned-scripts

# Verify
git log --oneline -3
```

`GIT_LFS_SKIP_SMUDGE=1` tells git to keep LFS objects as pointer files rather than trying to download the actual content. This is the right behavior for a training machine — we only need the Python/YAML code, not the 500 MB of doc images.

## Do Not
- Never run `git pull` on a fresh studio without first checking `git status` for LFS issues
- Never run `git add .` on a studio — it will stage hundreds of large binary files as "modifications"

## Do
- Always use `GIT_LFS_SKIP_SMUDGE=1` for any git operation on training machines
- After `git reset --hard`, verify with `git log --oneline -3` that HEAD matches the expected commit

## Related
- [[Lightning Studio Environment]]
- [[SSH Key Recovery]]
