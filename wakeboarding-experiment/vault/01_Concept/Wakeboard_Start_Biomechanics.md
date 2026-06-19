---
tags: [concept, biomechanics]
---

# Wakeboard Start Biomechanics (the 5 rules)

Why beginners fail the **deep-water start**: the boat yanks them to ~30 km/h and they either stand too fast, pull the rope, or let the board dig in → face-plant. The correct start is counter-intuitive: **stay small and let the pull lift you.** These 5 rules become reward terms ([[Reward_Design]]).

| # | Rule | What it means | Becomes reward |
|---|---|---|---|
| 1 | **Stay crouched, rise gradually** | knees to chest at start; extend legs slowly as you plane up — never jump to your feet | `height_progress` (phase-gated) + `stand_too_fast` penalty |
| 2 | **Positive board angle ≥10°** | weight the heels so the board tips up ~10–20° and planes to the surface instead of digging in | `board_positive_angle` |
| 3 | **Arms straight, handle at hips** | keep elbows extended, handle low near the hips — let the rope be a tow line, not a pull-up bar | `arms_straight` + `handle_at_hips` |
| 4 | **Don't pull against the rope** | bending the elbows to pull yourself up fights the boat and throws you forward → the #1 face-plant cause | `pull_against_rope` penalty |
| 5 | **Lean slightly back, knees soft** | small backward lean, knees never locked, weight balanced both feet | `lean_back_moderate` + `knee_bend_maintained` |

## The phase structure (maps to the policy's phase clock)
1. **Cannonball / float** — compact crouch, board perpendicular, tip up.
2. **Plane-up** — boat pulls; heels weighted, board planes; body still compact.
3. **Gradual rise** — legs extend slowly, handle stays at hips.
4. **Stable ride** — tall-ish crouch, balanced, board tracking forward.

## Sources
- [Monster Tower — getting up & staying up](https://monstertower.com/blog/post/getting-up-staying-up-on-a-wakeboard)
- [WaterVolleyball — arms straight, handle at hips](https://watervolleyball.com/wakeboarding-for-beginners/)
- [Axis — body position](https://blog.axiswake.com/beginner-wakeboarding-tips)
- [Monster Tower — proper positioning](https://monstertower.com/blog/post/proper-positioning-for-wakeboarding)

Related: [[Reward_Design]] · [[Environment_and_Rope_Model]] · [[RL_Method_HumanUP_AMP]]
