---
tags: [failure, navigation, command-conditioning, fundamental]
---

# Decorative Navigation Defect

## The Original Problem in the Repo

The original humanoid repo had a "command-conditioned navigation" task that was **fake**:
- The velocity command was a fixed hash ("walk forward")
- Markers had fixed positions (not randomized per episode)  
- Reward = plain velocity tracking (nothing to do with which marker)
- The language/colored markers were **decorative** — behavior didn't depend on them at all

The robot would walk and the demo looked reasonable, but it ignored its inputs entirely.

---

## Why This Is a Fundamental ML Problem

This isn't just a coding bug — it's a failure of experimental design. The model looked like it was navigating but was actually ignoring the conditioning signal. You can't tell by watching the demo; you have to **probe it**.

The probe: change the command while keeping everything else the same. If behavior doesn't change → the model is ignoring the input. This applies to any conditional model (language models, conditional diffusion, etc.).

---

## The Fix

Built genuine command-conditioned tasks from scratch:
1. **Randomize per episode** — different marker positions, different commanded target every episode
2. **Command-conditioned reward** — `commanded_target_reward` specifically rewards approaching the *commanded* marker and penalizes approaching wrong markers
3. **Instruction-swap probe** — verified that `success_by_command = [95.8%, 93.4%]` for different commands, confirming the behavior changes

Result: the same robot, the same locomotion controller, but now genuinely navigating to whatever colored marker it's told to go to — 94.5% success rate.

---

## The General Lesson

> **"Looks like it works" ≠ "works."** Always verify that your model's behavior is causally dependent on its conditioning inputs.

For RL nav tasks: run an instruction-swap evaluation. For language models: test with adversarial inputs. For conditional generation: check that conditioning is actually used (classifier-free guidance scale matters).

---

## Related

- [[Command-Conditioned Navigation]]
- [[P0 - CommandNav]]
- [[00 - Failure Index]]
