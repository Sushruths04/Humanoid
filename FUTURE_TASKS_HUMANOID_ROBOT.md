# Future Tasks for Humanoid Robot Training

This plan extends the current G1 humanoid work from stable locomotion and marker navigation into advanced language-conditioned humanoid control. The goal is to move step by step toward complex embodied behavior without jumping into a task that is too hard to debug.

## Current Stage

The current system is assumed to have:

- G1 baseline locomotion training working.
- Marker-based navigation working or producing good early results.
- Isaac Lab environment running on a GPU machine.
- Thesis pipeline scripts available for repeatable experiments.

The next work should focus on harder tasks that teach useful robotics concepts:

- Command-conditioned control.
- Multi-goal navigation.
- Sequential instruction following.
- Obstacle avoidance.
- Robust balance recovery.
- Vision-based navigation.

## Recommended Roadmap

## Phase 1: Commanded Locomotion

Train the humanoid to follow simple motion commands.

Example commands:

- Walk forward.
- Walk backward.
- Turn left.
- Turn right.
- Move slowly.
- Move quickly.
- Stop.

Why this matters:

This is the foundation for language-conditioned humanoid control. Before the robot can follow complex instructions, it must reliably map simple commands to different motion behaviors.

Implementation idea:

- Add a command input to the policy observation.
- Randomize the command during training.
- Reward velocity tracking, heading control, stability, and energy efficiency.

Success metrics:

- Target velocity tracking error.
- Fall rate.
- Episode length.
- Smoothness of motion.
- Ability to stop without drifting.

## Phase 2: Multi-Goal Marker Navigation

Extend the current red/blue marker task into several possible goals.

Example commands:

- Go to red.
- Go to blue.
- Go to green.
- Go to yellow.
- Go to the nearest marker.
- Go to the farthest marker.

Why this matters:

This turns the task from simple navigation into instruction grounding. The robot must understand which target the command refers to.

Implementation idea:

- Spawn 3-5 colored markers in random positions.
- Provide target identity through a command embedding or one-hot vector.
- Reward progress toward the correct marker.
- Penalize moving toward the wrong marker.

Success metrics:

- Correct target reach rate.
- Wrong target approach rate.
- Average distance to commanded marker.
- Time to reach target.

## Phase 3: Sequential Instruction Following

Train the humanoid to complete multiple goals in order.

Example commands:

- Go to red, then blue.
- Go to green, stop, then turn around.
- Walk to blue, wait, then return home.
- Visit red, blue, and green in order.

Why this matters:

Sequential tasks introduce memory, phase tracking, and long-horizon reward design. This is a major step toward real VLA-style robot behavior.

Implementation idea:

- Track the current subgoal in the environment.
- Update the active target after the robot reaches each marker.
- Add a phase variable to the observation.
- Start with two-step tasks, then move to three-step tasks.

Success metrics:

- Full sequence completion rate.
- Correct ordering rate.
- Average number of completed subgoals.
- Failure mode breakdown: fall, wrong target, timeout, instability.

## Phase 4: Obstacle Avoidance

Add walls, boxes, or barriers between the humanoid and the goal.

Example commands:

- Go to red without hitting obstacles.
- Walk around the wall.
- Reach the marker through the gap.
- Go to blue while avoiding boxes.

Why this matters:

This forces the policy to combine locomotion, navigation, and environment awareness. It is more realistic than open-space marker navigation.

Implementation idea:

- Start with static obstacles and wide gaps.
- Gradually randomize obstacle positions.
- Penalize collisions.
- Reward progress toward the target.
- Use curriculum learning: easy layouts first, harder layouts later.

Success metrics:

- Goal reach rate.
- Collision rate.
- Average path efficiency.
- Fall rate near obstacles.

## Phase 5: Push Recovery and Balance Robustness

Apply random external forces while the humanoid stands, walks, or navigates.

Example tasks:

- Stand still after being pushed.
- Keep walking after a disturbance.
- Recover balance and continue to target.
- Complete marker navigation while random pushes occur.

Why this matters:

Push recovery is a serious humanoid benchmark. A policy that can recover from disturbances is much stronger than one that only works in perfect conditions.

Implementation idea:

- Apply random forces to the torso at random intervals.
- Start with small pushes.
- Increase force magnitude over training.
- Reward not falling, returning to target velocity, and continuing the task.

Success metrics:

- Recovery rate after push.
- Time to recover stable walking.
- Fall rate under disturbance.
- Navigation success under disturbance.

## Phase 6: Language + Style Control

Add style or behavior modifiers to the commands.

Example commands:

- Walk slowly to red.
- Walk quickly to blue.
- Move carefully around the obstacle.
- Stop near the marker.
- Turn sharply.

Why this matters:

This goes beyond choosing a target. The robot must adapt how it moves based on language-like instructions.

Implementation idea:

- Add style parameters such as desired speed, stopping distance, turn sharpness, or energy preference.
- Use command embeddings that combine goal identity and style.
- Reward both task success and matching the requested style.

Success metrics:

- Goal success rate.
- Speed tracking error.
- Stop distance accuracy.
- Smoothness and stability.

## Phase 7: Long-Horizon Patrol Task

Create a small arena where the robot must visit several checkpoints.

Example commands:

- Patrol red-blue-green.
- Visit all markers.
- Visit red and blue, then return home.
- Complete a full inspection route.

Why this matters:

This tests long-horizon control, sequence memory, and robustness over longer episodes.

Implementation idea:

- Define a route as an ordered list of targets.
- Give reward for each completed checkpoint.
- Add timeout penalties.
- Add curriculum from two checkpoints to five checkpoints.

Success metrics:

- Full route completion rate.
- Average checkpoints completed.
- Total episode distance.
- Stability over long rollouts.

## Phase 8: Simple Object Interaction

Introduce contact-rich tasks with simple objects.

Example tasks:

- Walk to the cube.
- Push the cube to the target.
- Kick the ball toward a goal.
- Stand near an object and face it.

Why this matters:

This moves the work from pure locomotion into embodied interaction. It is harder because contact dynamics can be unstable.

Implementation idea:

- Start with large, lightweight objects.
- Use simple rewards for object displacement toward a target.
- Keep the first version non-language-conditioned.
- Add language commands only after the physical task works.

Success metrics:

- Object reaches target.
- Robot remains standing.
- Contact stability.
- Number of successful pushes or kicks.

## Phase 9: Vision-Based Marker Navigation

Replace privileged marker position inputs with camera observations.

Example task:

- Use onboard camera input to find the colored marker.
- Walk toward the commanded marker.
- Avoid obstacles using visual information.

Why this matters:

This is closer to real robot learning and VLA research. It is significantly harder than state-based control, so it should come after the state-based tasks are stable.

Implementation idea:

- Start with one front-facing camera.
- Use low-resolution RGB or segmentation input.
- First train/evaluate perception separately if needed.
- Consider using privileged state for teacher policy and vision for student policy.

Success metrics:

- Visual target detection success.
- Navigation success from camera input.
- Robustness to lighting and marker placement.
- Performance gap between state-based and vision-based policies.

## Best Next Complex Task

The best advanced task to work on next is:

Language-conditioned sequential navigation with obstacles.

Target behavior:

The humanoid receives a command such as:

```text
Go to the red marker, avoid obstacles, then go to the blue marker and stop.
```

This task is strong because it combines:

- Language-conditioned target selection.
- Sequential goal completion.
- Humanoid locomotion.
- Obstacle avoidance.
- Long-horizon reward design.
- Stopping behavior.

Suggested staged version:

1. Red then blue, no obstacles.
2. Red then blue with simple obstacles.
3. Random two-marker sequence with simple obstacles.
4. Random three-marker sequence with obstacles.
5. Add style modifiers such as slow, fast, or careful.

## Experiment Tracking

For each task, save:

- Task name.
- Command set.
- Reward terms.
- Observation terms.
- Training iterations.
- GPU type.
- Isaac Lab commit or version.
- Warp version.
- Success rate.
- Fall rate.
- Best checkpoint path.
- Short video or rollout GIF.

Recommended result table:

| Task | Success Rate | Fall Rate | Avg Episode Length | Best Checkpoint | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| G1 baseline walk | TBD | TBD | TBD | TBD | Current baseline |
| Red/blue navigation | TBD | TBD | TBD | TBD | Current marker task |
| Commanded locomotion | TBD | TBD | TBD | TBD | Next foundation task |
| Sequential red-blue | TBD | TBD | TBD | TBD | First advanced task |
| Obstacle red-blue | TBD | TBD | TBD | TBD | Recommended thesis extension |

## Practical Priority Order

Use this order if the goal is to reach an advanced but realistic thesis-level result:

1. Commanded locomotion.
2. Multi-goal marker navigation.
3. Sequential red-then-blue navigation.
4. Sequential navigation with obstacles.
5. Push recovery during navigation.
6. Language and style-conditioned navigation.
7. Vision-based navigation.
8. Object interaction.

## Final Recommendation

Do not jump directly to vision or object manipulation. The best next step is sequential language-conditioned navigation with obstacles, because it is advanced enough to be meaningful but still close to the current marker-navigation work.

The strongest thesis direction is:

```text
Language-conditioned G1 humanoid navigation with sequential goals, obstacle avoidance, and robustness testing.
```

