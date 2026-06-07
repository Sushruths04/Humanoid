---
tags: [task, c5, codex, agent-brief, capstone]
---

# C5 — Codex Agent Brief: Language-Driven Loco-Manipulation Capstone

> Copy-paste this entire document as the prompt for your Codex agent. It is self-contained.

---

## Context

You are implementing **C5 — the capstone loco-manipulation task** for a Physical-AI thesis project. This task integrates previously-trained components into a single system where a humanoid robot (NVIDIA Isaac Lab G1) receives a natural-language instruction like *"Walk to the table and put the red block on the plate"*, decomposes it into subgoals, and executes them using existing nav + manipulation policies with world-model lookahead.

**Repository**: `feat/planned-scripts` branch of `https://github.com/mitvho09/humanoid-g1-nav`  
**HuggingFace**: `mitvho09/humanoid-g1-nav` (checkpoints and videos)  
**Working dir on remote**: `/teamspace/studios/this_studio/Humanoid`  
**GPU target**: L40S 48 GB (Lightning AI Studio)

---

## Previously-Completed Components (reuse, do not retrain)

| Component | Location | What it does |
|---|---|---|
| P3 nav policy | `checkpoints/p3_vision_nav/model_499.pt` | Vision-based locomotion; reaches goal from RGB obs |
| T1 GR00T LoRA | `checkpoints/t1_groot/lora_final.pt` | GR00T N1.7 fine-tuned for tabletop manipulation |
| P4 Cosmos LoRA | `checkpoints/p4_cosmos/lora_step0500.pt` | Action-conditioned world model (Cosmos Predict2 2B) |
| P3 env | `programs/p3_vision_nav/` | Isaac Lab env: G1 walking, RGB obs, vel commands |
| T1 env | `programs/t1_groot_lora/` | Isaac Lab env: G1 upper body, tabletop objects |

---

## Architecture

```
NL instruction (user string)
        |
        v
[CPC5.2] Skill Router  (rule-based regex/keyword OR Cosmos-Reason if available)
        |  emits: [(NAV, "table"), (MANIP, "pick red block"), (MANIP, "place on plate")]
        v
[CPC5.3] Staged Hand-off Controller
        |  for each subgoal:
        |    if NAV:  run P3 nav policy until arrival condition met
        |    if MANIP: run T1 GR00T policy until grasp/place success
        v
[CPC5.4] World Model Veto (optional at each step)
        |  Cosmos lookahead: predict next 4 frames
        |  if predicted failure: re-approach or reroute
        v
[CPC5.1] Unified Isaac Lab Env
        |  G1 with both locomotion AND arm joints enabled
        v
[CPC5.5] Eval: Isaac Lab Arena + success logging
[CPC5.6] Video render + landing page
```

**Key design principle**: Use staged hand-off (stop walking → switch to manipulation), NOT simultaneous whole-body. Whole-body is stretch goal only.

---

## Checkpoint Implementation Plan

---

### CPC5.1 — Unified Embodiment

**Goal**: One Isaac Lab environment where G1 can locomote AND actuate arms in the same episode.

**Files to create**:
- `programs/c5_capstone/unified_env.py` — the unified env
- `programs/c5_capstone/test_unified_env.py` — smoke test

**Implementation**:

```python
# programs/c5_capstone/unified_env.py
import torch
import isaaclab.envs as envs
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

class UnifiedG1Env:
    """G1 with locomotion + arm joints. Use staged mode: one mode active at a time."""

    # Nav action dim: 3 (vx, vy, vyaw)  -- same as P3
    NAV_ACTION_DIM = 3
    # Manip action dim: 7 (7-DOF arm)   -- same as T1
    MANIP_ACTION_DIM = 7
    # Mode: "nav" or "manip"
    
    def __init__(self, cfg_path: str, num_envs: int = 1, device: str = "cuda"):
        # Load unified config that has both locomotion joints AND arm joints
        # Base: programs/p3_vision_nav/env_cfg.py  (locomotion)
        # Add: arm joint asset refs from programs/t1_groot_lora/env_cfg.py
        from programs.p3_vision_nav.env_cfg import G1VisionNavEnvCfg
        cfg = G1VisionNavEnvCfg()
        cfg.scene.num_envs = num_envs
        # Enable arm joints (add to articulation asset)
        cfg.scene.robot.spawn.usd_path = "assets/g1_with_arms.usd"
        cfg.scene.robot.actuators["arms"] = ...  # add arm actuators
        self.env = envs.ManagerBasedRLEnv(cfg=cfg)
        self.mode = "nav"
    
    def set_mode(self, mode: str):
        """Switch between 'nav' and 'manip' modes."""
        assert mode in ("nav", "manip")
        self.mode = mode
    
    def step(self, action: torch.Tensor):
        """action shape depends on mode: [B, 3] for nav, [B, 7] for manip."""
        full_action = torch.zeros(self.env.num_envs, self.env.action_space.shape[-1])
        if self.mode == "nav":
            full_action[:, :3] = action   # locomotion joints
        else:
            full_action[:, 3:10] = action  # arm joints
        return self.env.step(full_action)
    
    def get_obs(self) -> dict:
        obs = self.env.observation_manager.compute()
        return {
            "rgb": obs.get("rgb"),           # [B, H, W, 3]
            "proprio": obs.get("policy"),    # [B, proprio_dim]
            "ee_pos": obs.get("ee_pos"),     # [B, 3] end-effector position
        }
    
    def reset(self):
        return self.env.reset()
```

**Smoke test** (run this to verify CPC5.1):
```bash
cd /teamspace/studios/this_studio/Humanoid
python programs/c5_capstone/test_unified_env.py
# Expected: "UnifiedG1Env: nav step OK, manip step OK, shapes: (1, 48, 84, 3) (1, 7)"
```

---

### CPC5.2 — Skill Router from Reasoning

**Goal**: Parse NL instruction into ordered list of tagged subgoals.

**Files to create**:
- `programs/c5_capstone/skill_router.py`
- `programs/c5_capstone/test_skill_router.py`

**Implementation** (rule-based first; upgrade to Cosmos-Reason if time permits):

```python
# programs/c5_capstone/skill_router.py
import re
from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Subgoal:
    skill: Literal["NAV", "MANIP"]
    target: str       # e.g. "table", "red block", "plate"
    action: str       # e.g. "navigate", "pick", "place"

NAV_VERBS = {"walk", "go", "navigate", "move", "approach", "reach"}
MANIP_VERBS = {"pick", "grab", "place", "put", "drop", "push", "lift", "move"}

def parse_instruction(instruction: str) -> List[Subgoal]:
    """Rule-based instruction decomposition."""
    instruction = instruction.lower().strip()
    subgoals = []
    
    # Split on connectors: "and", "then", "after that", comma
    clauses = re.split(r"\band\b|\bthen\b|\bafter that\b|,", instruction)
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        
        words = clause.split()
        verb = words[0] if words else ""
        rest = " ".join(words[1:])
        
        # Extract object: strip prepositions
        obj = re.sub(r"^(to|the|a|an|onto|on|at)\s+", "", rest).strip()
        
        if verb in NAV_VERBS:
            subgoals.append(Subgoal(skill="NAV", target=obj, action=verb))
        elif verb in MANIP_VERBS:
            subgoals.append(Subgoal(skill="MANIP", target=obj, action=verb))
        else:
            # Default: if "to" present, probably nav; else manip
            if " to " in clause or clause.startswith("go"):
                subgoals.append(Subgoal(skill="NAV", target=obj, action="navigate"))
            else:
                subgoals.append(Subgoal(skill="MANIP", target=obj, action="manipulate"))
    
    return subgoals

# Example:
# parse_instruction("Walk to the table and put the red block on the plate")
# -> [Subgoal(NAV, "table", "walk"),
#     Subgoal(MANIP, "red block", "put"),   # NOTE: "on the plate" is the destination
#     ]
```

**10 test cases** (verify all decompose correctly):
```python
# programs/c5_capstone/test_skill_router.py
TESTS = [
    ("Walk to the table", [("NAV", "table")]),
    ("Go to the shelf and pick the red block", [("NAV", "shelf"), ("MANIP", "red block")]),
    ("Navigate to the door", [("NAV", "door")]),
    ("Pick up the cup", [("MANIP", "cup")]),
    ("Walk to the table and put the red block on the plate", [("NAV", "table"), ("MANIP", "red block")]),
    ("Move to the workbench then grab the screwdriver", [("NAV", "workbench"), ("MANIP", "screwdriver")]),
    ("Approach the box and push it", [("NAV", "box"), ("MANIP", "it")]),
    ("Lift the block", [("MANIP", "block")]),
    ("Go to the table, pick the apple, go to the plate, place the apple", [("NAV","table"),("MANIP","apple"),("NAV","plate"),("MANIP","apple")]),
    ("Navigate to charging dock", [("NAV", "charging dock")]),
]

def test_all():
    from skill_router import parse_instruction
    passed = 0
    for instr, expected in TESTS:
        result = parse_instruction(instr)
        skills = [(sg.skill, sg.target) for sg in result]
        if skills[:len(expected)] == expected:
            passed += 1
            print(f"PASS: {instr[:40]}")
        else:
            print(f"FAIL: {instr[:40]}")
            print(f"  expected: {expected}")
            print(f"  got:      {skills}")
    print(f"\n{passed}/10 passed")

if __name__ == "__main__":
    test_all()
```

---

### CPC5.3 — Staged Hand-off (Core Target)

**Goal**: Execute `navigate → arrive at table → switch to manipulation → pick → place` as staged sequence.

**Files to create**:
- `programs/c5_capstone/staged_controller.py`
- `programs/c5_capstone/run_c5.py`

**Load P3 nav policy**:
```python
# programs/c5_capstone/staged_controller.py
import torch
import numpy as np
from programs.p3_vision_nav.model import VisionNavPolicy   # reuse P3 policy class

def load_nav_policy(ckpt_path: str, device: str = "cuda") -> VisionNavPolicy:
    policy = VisionNavPolicy(obs_dim=48*84*3, act_dim=3)  # RGB flattened, 3-DOF vel
    state = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    return policy.to(device)
```

**Load T1 GR00T manipulation policy**:
```python
from gr00t.model.policy import Gr00tPolicy
from peft import PeftModel

def load_manip_policy(base_model_id: str, lora_ckpt: str, device: str = "cuda"):
    # GR00T N1.7 with LoRA
    policy = Gr00tPolicy.from_pretrained(base_model_id)
    policy = PeftModel.from_pretrained(policy, lora_ckpt)
    policy.eval()
    return policy.to(device)
```

**Arrival condition** (when to switch from NAV to MANIP):
```python
def at_target(obs: dict, target_name: str, threshold_m: float = 0.5) -> bool:
    """Check if robot is within threshold meters of target object."""
    # Use ee_pos from obs or object detection on RGB
    # Simple: check if object bounding box is large enough in RGB frame
    rgb = obs["rgb"][0]  # [H, W, 3]
    # Object detector or hardcoded color segmentation for demo
    # For demo: use proprio distance if available, else pixel-area heuristic
    if "obj_dist" in obs:
        return float(obs["obj_dist"][0]) < threshold_m
    # Pixel-area heuristic: target fills >5% of frame
    return False  # fallback: manual trigger
```

**Staged Controller**:
```python
class StagedController:
    def __init__(self, nav_policy, manip_policy, env, device="cuda"):
        self.nav_policy = nav_policy
        self.manip_policy = manip_policy
        self.env = env
        self.device = device
    
    def run_episode(self, subgoals: list, max_steps_per_subgoal: int = 500) -> dict:
        obs = self.env.reset()
        results = []
        
        for sg in subgoals:
            print(f"[C5] Executing: {sg.skill} -> {sg.target}")
            self.env.set_mode(sg.skill.lower())
            success = False
            
            for step in range(max_steps_per_subgoal):
                if sg.skill == "NAV":
                    # P3 policy: obs = RGB image
                    rgb = obs["rgb"].to(self.device).float() / 255.0
                    rgb_flat = rgb.view(rgb.shape[0], -1)
                    with torch.no_grad():
                        action = self.nav_policy(rgb_flat)
                    obs, reward, done, info = self.env.step(action)
                    
                    if at_target(obs, sg.target):
                        success = True
                        break
                
                elif sg.skill == "MANIP":
                    # T1 GR00T policy
                    rgb = obs["rgb"]
                    proprio = obs["proprio"]
                    with torch.no_grad():
                        action = self.manip_policy(rgb=rgb, state=proprio)
                    obs, reward, done, info = self.env.step(action)
                    
                    if float(reward[0]) > 0.5:  # manipulation reward threshold
                        success = True
                        break
            
            results.append({"subgoal": sg, "success": success, "steps": step + 1})
            if not success:
                print(f"[C5] WARNING: subgoal failed: {sg.skill} {sg.target}")
        
        return {
            "subgoals": results,
            "full_success": all(r["success"] for r in results),
        }
```

**Run script** (`programs/c5_capstone/run_c5.py`):
```python
# programs/c5_capstone/run_c5.py
import argparse
import torch
import mediapy
from skill_router import parse_instruction
from unified_env import UnifiedG1Env
from staged_controller import StagedController, load_nav_policy, load_manip_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str,
        default="Walk to the table and put the red block on the plate")
    parser.add_argument("--nav-ckpt", type=str,
        default="checkpoints/p3_vision_nav/model_499.pt")
    parser.add_argument("--manip-ckpt", type=str,
        default="checkpoints/t1_groot/lora_final.pt")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--out", type=str, default="docs/results/c5_demo.mp4")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[C5] Instruction: {args.instruction}")
    subgoals = parse_instruction(args.instruction)
    print(f"[C5] Decomposed: {[(sg.skill, sg.target) for sg in subgoals]}")
    
    env = UnifiedG1Env(num_envs=1, device=device)
    nav_policy = load_nav_policy(args.nav_ckpt, device)
    manip_policy = load_manip_policy("nvidia/GR00T-N1.5-3B", args.manip_ckpt, device)
    
    controller = StagedController(nav_policy, manip_policy, env, device)
    
    successes = 0
    frames = []
    
    for trial in range(args.num_trials):
        result = controller.run_episode(subgoals)
        if result["full_success"]:
            successes += 1
        print(f"Trial {trial+1}/{args.num_trials}: {'SUCCESS' if result['full_success'] else 'FAIL'}")
        # Collect frames for video
        frames.extend(result.get("frames", []))
    
    print(f"\n[C5] CPC5.3 Result: {successes}/{args.num_trials} full-sequence successes")
    
    if frames:
        mediapy.write_video(args.out, frames, fps=30)
        print(f"[C5] Video saved: {args.out}")

if __name__ == "__main__":
    main()
```

---

### CPC5.4 — World Model in the Loop

**Goal**: Insert Cosmos lookahead at decision points. Veto bad actions; documented case where it changes outcome.

**Files to create**:
- `programs/c5_capstone/world_model_veto.py`

```python
# programs/c5_capstone/world_model_veto.py
import torch
import numpy as np
from cosmos_predict2.pipelines.video2world import Video2WorldActionConditionedPipeline
from cosmos_predict2.conditioner import ActionCondition, DataType
from peft import inject_adapter_in_model, LoraConfig

class WorldModelVeto:
    """Use P4 Cosmos world model to predict next frames and veto bad actions."""
    
    LORA_TARGETS = ["q_proj","k_proj","v_proj","output_proj","mlp.layer1","mlp.layer2"]
    
    def __init__(self, cosmos_dir: str, lora_ckpt: str, device: str = "cuda"):
        self.device = device
        self.pipe = Video2WorldActionConditionedPipeline.from_config(
            config_job_name="Cosmos-Predict2-2B-Video2World-Sample-AV",
            checkpoint_dir=cosmos_dir,
            checkpoint_name="model.pt",
            precision="bf16",
            device=device,
        )
        # Load LoRA
        lora_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                              target_modules=self.LORA_TARGETS, bias="none")
        inject_adapter_in_model(lora_cfg, self.pipe.model)
        state = torch.load(lora_ckpt, map_location=device)
        self.pipe.model.load_state_dict(state, strict=False)
        self.pipe.model.eval()
    
    def should_veto(self, rgb_t: torch.Tensor, action: torch.Tensor,
                    failure_threshold: float = 0.3) -> bool:
        """
        Predict next frame given current obs + action.
        Veto if predicted frame deviates too much from expected (high noise = collision/failure).
        
        Args:
            rgb_t: [1, 3, T, H, W] current video window (T=9 for (T-1)%4==0)
            action: [1, 12, 7] action sequence
            failure_threshold: latent noise variance above this -> veto
        """
        vid = rgb_t.to(self.device).to(torch.bfloat16) / 127.5 - 1.0
        act = action.to(self.device).to(torch.bfloat16)
        
        with torch.no_grad():
            latent = self.pipe.tokenizer.encode(vid)
            B, C, T_lat, H_lat, W_lat = latent.shape
            
            mask = torch.zeros(B, 1, T_lat, H_lat, W_lat,
                               device=self.device, dtype=torch.bfloat16)
            mask[:, :, :1] = 1.0
            
            condition = ActionCondition(
                crossattn_emb=torch.zeros(B, 256, 1024,
                                          device=self.device, dtype=torch.bfloat16),
                data_type=DataType.VIDEO,
                padding_mask=torch.zeros(B, 1, vid.shape[3], vid.shape[4],
                                          device=self.device, dtype=torch.bfloat16),
                fps=None, use_video_condition=True,
                gt_frames=latent,
                condition_video_input_mask_B_C_T_H_W=mask,
                action=act,
            )
            
            # Single-step prediction at high sigma for speed
            sigma_t = torch.full((B,), 1.0, device=self.device, dtype=torch.bfloat16)
            noise = torch.randn_like(latent)
            pred = self.pipe.denoise(noise, sigma_t, condition)
            
            # High residual = model uncertain = likely failure
            residual_var = float(torch.var(pred.x0 - latent).cpu())
        
        return residual_var > failure_threshold
```

**Integration into StagedController**: In `run_episode`, before executing each action:
```python
# In StagedController.run_episode, add veto check:
if self.wm_veto and sg.skill == "MANIP":
    rgb_window = collect_last_9_frames(obs)  # T=9 satisfies (T-1)%4==0
    if self.wm_veto.should_veto(rgb_window, action):
        print("[C5] World model veto: re-planning action")
        action = self.replan_action(obs)  # sample alternative
```

---

### CPC5.5 — Benchmark + Photoreal Eval

**Scenarios to evaluate** (minimum 2 for DoD):

| Scenario | Nav target | Manipulation |
|---|---|---|
| A | "Walk to table, pick red block, place on plate" | table -> pick -> plate |
| B | "Go to shelf and grab the cup" | shelf -> pick |
| C | "Navigate to workbench and push the box off" | workbench -> push |

**Run command**:
```bash
python programs/c5_capstone/run_c5.py \
  --instruction "Walk to the table and put the red block on the plate" \
  --num-trials 10 \
  --out docs/results/c5_scenario_a.mp4

python programs/c5_capstone/run_c5.py \
  --instruction "Go to the shelf and grab the cup" \
  --num-trials 10 \
  --out docs/results/c5_scenario_b.mp4
```

**Expected results table** (fill in after running):
```
| Scenario | Full success | Nav only | Manip only |
|---|---|---|---|
| A | X/10 | Y/10 | Z/10 |
| B | X/10 | Y/10 | Z/10 |
```

**Photoreal eval** (if Cosmos-Transfer available):
```bash
# Generate photoreal versions of success episodes
python programs/c5_capstone/photoreal_render.py \
  --video docs/results/c5_scenario_a.mp4 \
  --out docs/results/c5_photoreal_a.mp4
```

---

### CPC5.6 — Flagship Cut

**Video composition**:
```bash
# Combine: canonical instruction text overlay + nav clip + manip clip + success
ffmpeg -i docs/results/c5_scenario_a.mp4 \
       -vf "drawtext=text='Walk to the table and put the red block on the plate':fontsize=24:x=10:y=10" \
       docs/results/c5_flagship.mp4
```

**Results doc** (`docs/results/c5_capstone.md`): fill in with final numbers.

**HuggingFace upload**:
```bash
export HF_TOKEN=<from environment, never hardcode>
python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
for f in ["docs/results/c5_flagship.mp4", "docs/results/c5_scenario_a.mp4", "docs/results/c5_scenario_b.mp4"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"videos/c5/{f.split('/')[-1]}",
                    repo_id="mitvho09/humanoid-g1-nav", token="$HF_TOKEN")
EOF
```

---

## Setup Commands (run first on Lightning AI L40S)

```bash
# 1. Environment
conda activate isaac_env  # or your Isaac Lab conda env

# 2. Install packages (apt not persistent — run every session)
sudo apt-get install -y ffmpeg

# 3. PYTHONPATH
export PYTHONPATH=/tmp/te_stub:/tmp/cosmos-predict2:$PYTHONPATH

# 4. Verify Cosmos available
python -c "from cosmos_predict2.pipelines.video2world import Video2WorldActionConditionedPipeline; print('OK')"

# 5. Verify P3 checkpoint exists
ls checkpoints/p3_vision_nav/model_499.pt

# 6. Verify T1 GR00T checkpoint exists
ls checkpoints/t1_groot/lora_final.pt

# 7. Verify P4 Cosmos LoRA exists
ls checkpoints/p4_cosmos/lora_step0500.pt
```

---

## Critical Rules (never violate)

1. **HF token**: Always use `$HF_TOKEN` env var. Never commit token to any file.
2. **Git push from remote only**: Always commit and push from Lightning AI machine, not local Windows (LFS mismatch).
3. **Branch**: Always push to `feat/planned-scripts`, never `main`.
4. **Cosmos temporal dim**: Video T must satisfy `(T-1) % 4 == 0`. Pad to 9 or 13.
5. **padding_mask**: Always pass explicit `torch.zeros(B, 1, H, W, dtype=bfloat16)`, never None.
6. **Smoke test first**: Before any long run, do 2-iter smoke test to catch API errors early.
7. **Save frames during eval**: Always write video while running, not after (process may crash).
8. **Graceful degradation**: If full loco-manip fails, deliver "stitched demo" — separate nav + manip clips unified by skill router. Document whole-body as future work.

---

## Deliverables Checklist

- [ ] `programs/c5_capstone/unified_env.py` — unified G1 env
- [ ] `programs/c5_capstone/skill_router.py` — NL -> subgoal decomposition
- [ ] `programs/c5_capstone/staged_controller.py` — nav/manip hand-off
- [ ] `programs/c5_capstone/world_model_veto.py` — Cosmos lookahead veto
- [ ] `programs/c5_capstone/run_c5.py` — end-to-end runner
- [ ] `docs/results/c5_scenario_a.mp4` + `c5_scenario_b.mp4` — eval videos
- [ ] `docs/results/c5_flagship.mp4` — polished demo video
- [ ] `docs/results/c5_capstone.md` — results table
- [ ] HF upload: `videos/c5/*.mp4`
- [ ] GitHub commit on `feat/planned-scripts`

---

## Related Vault Docs

- [[C5 - Loco-Manipulation Capstone]] — original task spec with DoD
- [[P3 - VisionNav]] — nav policy source
- [[T1 - GR00T LoRA]] — manipulation policy source
- [[P4 - Cosmos Predict Results]] — world model (Cosmos) API patterns
- [[P4 - Cosmos Failures and Lessons]] — 9 API gotchas to avoid
