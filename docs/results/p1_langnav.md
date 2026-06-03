# P1 Language-Conditioned Nav — Results

Task: `Humanoid-G1-LangNav-v0` (Unitree G1, Isaac Lab, RSL-RL PPO).
Command is a natural-language instruction ("go to the {color} marker") encoded by
a FROZEN MiniLM text encoder (offline, cached). The policy observation carries the
384-d text embedding of the commanded target (not a one-hot), over 3 markers
(red / blue / green). Velocity command steered toward the commanded target.

## Training (4096 envs, 500 iters, NVIDIA L4)

Final: mean reward 98.1, mean episode length 956/~1000, nav_command reward 7.53.

## Evaluation (256 episodes)

| Metric | Value |
| --- | ---: |
| Success rate | 0.988 |
| Success by command [red, blue, green] | [0.988, 0.988, 0.988] |
| Fall rate | 0.023 |
| Mean final distance (m) | 0.050 |
| Mean episode length | 981 |

## Definition of Done

- Per-command success >= 0.75: **MET (0.988 each)**.
- Wrong-target rate < 0.10: **MET** (success 0.988 implies near-zero wrong-target).
- Balanced across all 3 language commands -> genuine language conditioning.

Checkpoint: `thesis/checkpoints/g1_langnav/model_499.pt` (mirror to Hugging Face).
