---
tags: [log, live]
---

# Experiment Log (append-only)

> Per [[DOC_PROTOCOL]]: one entry per training run. Newest on top. Template below.

## Template
```
### YYYY-MM-DD — <stage> — <short title>
- Config: configs/<file>.yaml   | Git: <commit>
- v_pull: <km/h>  | envs: <N>  | iters: <N>  | hardware: <L40S/...>
- Checkpoint: <path>  (HF: <link>)
- Metrics: success <%>, fall <%>, time-to-stand <s>, board-angle <%>, smoothness <..>
- Observation: <one line — what happened / what to change next>
```

---

### 2026-06-19 — plan — project created
- Plan written (`PLAN.md`), vault scaffolded. No runs yet.
- Next: PLAN §14 task 1 (scaffold) → task 2 (board + foot binding).
