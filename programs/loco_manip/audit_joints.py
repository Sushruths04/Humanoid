"""Audit G1 joint structure without launching Isaac Sim.

Run on Lightning AI:
  cd /teamspace/studios/this_studio/repo
  PYTHONPATH=. python programs/loco_manip/audit_joints.py
"""
import torch
import sys, os
sys.path.insert(0, os.path.abspath("."))

CHECKPOINT = "checkpoints/g1_commandnav_stable/model_499.pt"


def audit_checkpoint():
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    actor = ckpt["actor_state_dict"]

    first_key = [k for k in actor if "mlp.0.weight" in k][0]
    last_key  = [k for k in actor if "mlp.6.weight" in k][0]
    obs_dim = actor[first_key].shape[1]
    act_dim = actor[last_key].shape[0]

    print(f"=== model_499.pt checkpoint audit ===")
    print(f"obs_dim : {obs_dim}   (127 = proprioception + nav command)")
    print(f"act_dim : {act_dim}   (37 = joint position targets)")
    print()
    print("All keys in actor_state_dict:")
    for k, v in actor.items():
        print(f"  {k:50s}  shape={list(v.shape)}")


def explain_obs_breakdown():
    print("""
=== What is inside the 127-dim observation? ===

Isaac Lab's G1FlatEnvCfg builds the observation from these groups:

  base_lin_vel        3   (x, y, z velocity of the pelvis)
  base_ang_vel        3   (roll, pitch, yaw rate of the pelvis)
  projected_gravity   3   (gravity vector in robot frame — tells robot if tilting)
  velocity_commands   3   (commanded vx, vy, yaw_rate from the command manager)
  joint_pos          37   (current joint positions minus default positions)
  joint_vel          37   (current joint velocities)
  actions            37   (last actions sent — helps policy be smooth)
  nav_command         4   (2-dim one-hot for which marker + 2-dim relative xy)
  ─────────────────────
  TOTAL             127
""")


def explain_action_breakdown():
    print("""
=== What is inside the 37-dim action? ===

The 37 actions are joint POSITION TARGETS (delta from default pose).
The robot's PD controller converts these to torques:
  torque = kp * (target - current_pos) + kd * (0 - current_vel)

Joint groups:
  legs (12):    left/right hip (3 DOF) + knee (1) + ankle (2) = 6 per side × 2
  torso (3):    waist yaw + waist roll + torso
  arms (14):    shoulder (3) + elbow (1) + wrist (3) = 7 per side × 2
  hands (8):    finger joints (G1-29DOF variant includes them)
  ─────────────
  Total = 12 + 3 + 14 + 8 = 37

The NAV POLICY controls ALL 37 joints but arms are at default pose target.
The arms are NOT frozen — they can drift — but the reward never cared about
what the arms did, so the policy learned to keep them near default.
""")


if __name__ == "__main__":
    audit_checkpoint()
    explain_obs_breakdown()
    explain_action_breakdown()
