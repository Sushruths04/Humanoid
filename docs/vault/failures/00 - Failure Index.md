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

## Lessons Distilled

1. **Watch the per-term reward breakdown**, not just total reward. Flat task term + rising total = reward farming.
2. **Make the first success easy** in multi-step tasks — curriculum, closer targets, whatever it takes.
3. **Bind mounts only** — never write important outputs to unmounted container paths.
4. **Never paste tokens into chat** — rotate immediately if you do, then make the resource public.
5. **scp for code transfer**, never SSH heredocs with complex content.
6. **Isaac Sim processes linger** — kill in-container, don't poll host pgrep.
7. **Behavior probe** > test accuracy — verify the policy CHANGES BEHAVIOR with the command.

---

## Related

- [[00 - START HERE]]
- [[Common Failure Patterns]]
