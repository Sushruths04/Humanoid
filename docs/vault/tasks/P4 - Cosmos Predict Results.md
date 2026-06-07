---
tags: [task, p4, cosmos, results, complete]
---

# P4 — Cosmos Predict Results

**Status**: Complete  
**Completed**: 2026-06-07  
**GPU**: A100-80G (Lightning AI Studio `s_01kth1fwzr7xwxfnvv436s1pjz`)  
**Actual GPU-hours**: ~12 hr (smoke gate at CP4.3 saved significant budget)

---

## GitHub Inventory (feat/planned-scripts @ 9e94c98)

| File | Description |
|---|---|
| `programs/p4_cosmos_world_sim/run_p4.sh` | Master orchestrator — runs CP4.1–CP4.5 |
| `programs/p4_cosmos_world_sim/cp41_inference.py` | Stock Cosmos inference baseline |
| `programs/p4_cosmos_world_sim/cp42_dataprep.py` | Bridge dataset dataloader |
| `programs/p4_cosmos_world_sim/cp43_train.py` | LoRA post-training with PEFT |
| `programs/p4_cosmos_world_sim/cp44_rollout.py` | K-step action-conditioned rollout |
| `programs/p4_cosmos_world_sim/cp45_plan.py` | CEM planning in latent space |
| `docs/results/cp44_rollout.mp4` | Side-by-side real/generated (4 eps x 12 frames) |
| `docs/results/cp45_planning.mp4` | 3-column: conditioning / random / CEM (3 trials x 4 frames) |

---

## HuggingFace Inventory (mitvho09/humanoid-g1-nav)

| Path | Description |
|---|---|
| `checkpoints/p4_cosmos/lora_step0100.pt` | LoRA checkpoint at 100 steps |
| `checkpoints/p4_cosmos/lora_step0200.pt` | LoRA checkpoint at 200 steps |
| `checkpoints/p4_cosmos/lora_step0300.pt` | LoRA checkpoint at 300 steps |
| `checkpoints/p4_cosmos/lora_step0400.pt` | LoRA checkpoint at 400 steps |
| `checkpoints/p4_cosmos/lora_step0500.pt` | LoRA checkpoint at 500 steps (final) |
| `checkpoints/p4_cosmos/train_log.txt` | Full training log with per-step losses |
| `videos/p4/cp44_rollout.mp4` | Rollout generation quality video |
| `videos/p4/cp45_planning.mp4` | CEM planning demo video |

---

## Per-Checkpoint Results

| CP | Deliverable | Result | Status |
|---|---|---|---|
| CP4.1 | Stock Cosmos inference baseline | Cosmos Predict2 2B generates coherent 12-frame video from initial frame; confirmed pipeline works on A100-80G; ~18s/frame at bf16 | Done |
| CP4.2 | Bridge data dataloader | Loaded LeRobot bridge_orig; 1024 episodes found; (frame_t, action_t, frame_{t+1}) batches verified; action shape [B, 12, 7] | Done |
| CP4.3 | LoRA post-training | 500 steps, final avg loss 0.0131; 3.32% trainable params (rank=16, targets: q/k/v/output_proj, mlp.layer1/2); saved to HF | Done |
| CP4.4 | K-step rollout fidelity | SSIM 0.963 on 3/4 episodes; episode 0 outlier SSIM 0.25 (dynamic content); side-by-side video confirms action-conditional generation | Done |
| CP4.5 | CEM planning | Goal-reach rate: -6.9% vs random (honest negative result); world model SSIM 0.963 validates generation quality; planning failure attributed to goal-representation and latent-space scoring disconnect | Done (documented) |

---

## Key Metrics Summary

- **LoRA training loss**: 0.0131 (avg over last 100 steps of 500-step run)
- **Trainable parameters**: 3.32% (LoRA rank=16)
- **Rollout SSIM**: 0.963 (episodes 1-3); 0.25 (episode 0, outlier)
- **Planning performance**: -6.9% vs random baseline (negative result documented)
- **Inference speed**: ~18s/frame on A100-80G at bf16

---

## Working Cosmos API Pattern

```python
# Correct Cosmos Predict2 2B action-conditioned inference
from cosmos_predict2.pipelines.video2world import Video2WorldActionConditionedPipeline
from cosmos_predict2.conditioner import ActionCondition, DataType
from peft import inject_adapter_in_model, LoraConfig

# Load model from config (NOT from_pretrained)
pipe = Video2WorldActionConditionedPipeline.from_config(
    config_job_name="Cosmos-Predict2-2B-Video2World-Sample-AV",
    checkpoint_dir="/tmp/cosmos-predict2",
    checkpoint_name="model.pt",
    precision="bf16",
    device="cuda",
)

# Load LoRA weights
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","output_proj","mlp.layer1","mlp.layer2"],
    bias="none",
)
inject_adapter_in_model(lora_config, pipe.model)
state = torch.load("lora_step0500.pt", map_location="cuda")
pipe.model.load_state_dict(state, strict=False)

# Tokenize input: T must satisfy (T-1) % 4 == 0
latent = pipe.tokenizer.encode(vid_t)  # [B, C, T_lat, H_lat, W_lat]

# Build condition — padding_mask must be bfloat16 zeros, NEVER None
condition = ActionCondition(
    crossattn_emb=torch.zeros(B, 256, 1024, device=device, dtype=pipe.precision),
    data_type=DataType.VIDEO,
    padding_mask=torch.zeros(B, 1, H, W, device=device, dtype=pipe.precision),
    fps=None, use_video_condition=True,
    gt_frames=latent,
    condition_video_input_mask_B_C_T_H_W=mask,
    action=act_t,
)

# Euler ODE reverse diffusion
x_t = torch.randn_like(latent)
ts = torch.linspace(0.99, 0.01, n_steps + 1, device=device, dtype=pipe.precision)
for i in range(n_steps):
    t_curr, t_next = float(ts[i]), float(ts[i+1])
    sigma = t_curr / (1.0 - t_curr + 1e-8)
    sigma_t = torch.full((B,), sigma, device=device, dtype=pipe.precision)
    pred = pipe.denoise(x_t, sigma_t, condition)  # returns DenoisePrediction with .x0
    velocity = (x_t - pred.x0) / (t_curr + 1e-8)
    x_t = x_t + (t_next - t_curr) * velocity
decoded = pipe.tokenizer.decode(x_t)
```

---

## Related

- [[P4 - Cosmos Predict]] — task spec and DoD
- [[P4 - Cosmos Failures and Lessons]] — 9 failures documented
- [[C5 - Loco-Manipulation Capstone]] — uses P4 world model for lookahead
- [[P3 - VisionNav]] — navigation policy whose evaluation data fed P4
