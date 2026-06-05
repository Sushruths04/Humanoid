---
tags: [reference, failures, patterns, debugging]
---

# Common Failure Patterns

Quick diagnostic table. When something goes wrong, find your pattern here.

## RL Training Failures

| Pattern | Tell | Check | Fix |
|---|---|---|---|
| Reward farming / stand-still | total reward rises; task-specific term flat near 0 | per-term episode breakdown in training log | make first success easier; closer targets; curriculum |
| Decorative conditioning | per-command success rates all identical | instruction-swap probe | add command-conditioned reward + penalty for wrong targets |
| Reward doesn't improve at all | total reward flat after 200+ iters | check env reset + step events firing | smoke test with SMOKE=1; check event signatures (env_ids required) |
| Very high fall rate | `base_contact` termination > 30% | not a training bug — G1 policy characteristic | add fall penalty or increase termination penalty weight |

## Infrastructure Failures

| Pattern | Tell | Fix |
|---|---|---|
| Results missing after train | exit 0, no file on host | check bind mounts; write to `programs/results/`, not `docs/results/` |
| Docker pull denied | `unauthorized` or `denied` | make GHCR package public; see [[GHCR Auth Denied]] |
| Wrong Python | `ImportError: No module named 'isaaclab'` from host | use `isaaclab.sh -p` inside container, not bare `python3` |
| Container won't start | compose error about missing image | make sure `docker pull` succeeded; check tag matches |
| Build required | `container.py` trying to rebuild from NGC | use `docker compose --no-build` directly |
| File corrupted mid-write | SyntaxError after SSH heredoc | use scp to transfer files; see [[SSH Heredoc Apostrophe Corruption]] |
| Permission denied on result file | container wrote root-owned file | `docker exec ... chown -R uid:gid /workspace/programs/results` |

## Evaluation Failures

| Pattern | Tell | Fix |
|---|---|---|
| `KeyError: class_name` in eval | loading checkpoint crashes | add `handle_deprecated_rsl_rl_cfg` before `OnPolicyRunner` |
| `AttributeError: _nav_target_ids` | wrong evaluator for task | use `evaluate_seq.py` for SeqNav; `evaluate.py` only for CommandNav/ObstacleNav/LangNav |
| Eval numbers very low despite good training | maybe wrong checkpoint path, or task randomizes on reset | snapshot first-episode layout before policy steps |
| Video never appears | render started but no mp4 | first frame takes 2+ min; check log for errors |

## Process Management

| Pattern | Tell | Fix |
|---|---|---|
| `pkill` on host doesn't stop Isaac Sim | process still in `docker ps` | kill inside container: `docker exec isaac-lab-base bash -lc "pkill -f custom_train.py"` |
| Wrapper hung waiting on process | while-pgrep loop never exits | kill the lingering process in-container first |
| SSH times out after long job | connection dropped | use `nohup ... &` and redirect to log file; reconnect and check log |

## The Debugging Mindset

1. Check the **per-term reward breakdown** before anything else.
2. Run a **behavioral probe** (instruction-swap, min-distance check) to understand what the policy is actually doing.
3. **Rule out code bugs systematically** — build a ground-truth diagnostic (call the function directly, check outputs) before spending GPU time on another training run.
4. **Don't fix what you can't measure** — add instrumentation first, hypothesize second.

## Related

- [[00 - Failure Index]]
- [[SeqNav Stand-Still Local Optimum]]
- [[Reward Shaping & Progress Rewards]]
