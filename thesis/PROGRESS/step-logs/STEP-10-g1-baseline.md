# STEP 10 G1 baseline command gate

_2026-05-30T20:49:06Z_

## Baseline command

```bash
cd /home/zeus/content/Humanoid/IsaacLab
python scripts/environments/list_envs.py | grep -i 'G1\|loco'
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --headless --num_envs 512 --max_iterations 300
```

Run this only after Isaac Sim/Isaac Lab GPU smoke test succeeds.

# STEP 10 G1 baseline training

_2026-05-30T20:58:45Z_

## Baseline training

- Task: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`
- Envs: 512
- Max Iters: 300

```bash
docker exec isaac-lab /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --headless --num_envs 512 --max_iterations 300
```

# STEP 10 G1 baseline training

_2026-05-30T21:07:55Z_

## Baseline training

- Task: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`
- Envs: 512
- Max Iters: 300

```bash
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --headless --num_envs 512 --max_iterations 300
```

# STEP 10 G1 baseline training

_2026-05-30T21:09:54Z_

## Baseline training

- Task: `Isaac-Velocity-Flat-G1-v0`
- Envs: 512
- Max Iters: 300

```bash
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 300
```

# STEP 10 G1 baseline training

_2026-05-30T21:27:33Z_

## Baseline training

- Task: `Isaac-Velocity-Flat-G1-v0`
- Envs: 512
- Max Iters: 300

```bash
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 300
```

