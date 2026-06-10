# Why Arms Were "Frozen" During Nav Training

## They Weren't Actually Frozen

The arms were NOT frozen in the nav training env — the policy could move them.
It just never learned to.

## Why PPO Leaves Them At Default

PPO maximizes expected reward. The nav reward was:
```
reward = + progress_toward_marker
         + upright_bonus
         - fall_penalty
         - joint_velocity_penalty    ← small negative for high joint velocities
```

The last term (`joint_velocity_penalty`) punishes unnecessary movement.
Moving arms doesn't help reach the marker. It adds noise and energy cost.
PPO learned: "keep arms still = less penalty = higher total reward."

## Why This Matters for Loco-Manipulation

To make the robot pick things up:
1. We can't just tell the nav policy "move your arms to grab the block"
2. The nav policy has NEVER seen a reward for arm movement
3. Its weights encode "stable walking" not "picking up objects"

The solution: **train a second policy** on a fixed-base arm env.
That policy gets rewarded ONLY for grasping and placing.
Then the state machine decides WHICH policy runs at each moment.

## The Stability Tradeoff

Running both policies simultaneously (whole-body) is hard:
- Arm movement shifts the center of mass → destabilizes walking
- The walking policy was trained with arms still — it expects that
- Training both together requires curriculum learning or careful reward shaping

For the staged approach: we STOP the legs first, THEN move the arms.
This avoids the coupling entirely. Trade-off: not as impressive visually,
but much simpler to train and works reliably.

The fully simultaneous version (arms + legs at same time) is the "stretch goal"
mentioned in the C5 plan — possible with whole-body RL, but needs ~50 GPU-hours.
