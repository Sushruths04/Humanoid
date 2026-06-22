"""Pose viewer — shows INITIAL (cannonball) and TARGET (riding) poses side by side.

Run BEFORE training to verify:
  1. Robot spawns in the correct sideways cannonball crouched position
  2. Board is under the feet (not inside them)
  3. Target pose looks like a real wakeboard riding stance

No checkpoint needed. Runs headless on GPU and saves PNG screenshots.

Usage (inside Isaac Lab container):
    python pose_viewer.py
    python pose_viewer.py --steps 120   # hold each pose for N sim steps before screenshot

Output files:
    pose_initial.png   — cannonball crouch (what robot starts as)
    pose_target.png    — riding stance (what we want it to learn)
    pose_comparison.txt — side-by-side joint angle table printed to terminal + saved
"""
from __future__ import annotations
import argparse, math, os

DEG = math.pi / 180.0


# ── INITIAL POSE (cannonball deep-water start) ─────────────────────────────
CANNONBALL_ROOT_Z = 0.50
CANNONBALL_ROOT_ROT = (0.7071, 0.0, 0.0, 0.7071)   # wxyz — sideways, facing +Y

CANNONBALL_JOINTS = {
    "left_hip_pitch_joint":      -0.80,   # deep hip flexion (knees to chest)
    "right_hip_pitch_joint":     -0.80,
    "left_knee_joint":            1.40,   # deep knee bend
    "right_knee_joint":           1.40,
    "left_ankle_pitch_joint":     0.30,   # feet relatively flat
    "right_ankle_pitch_joint":    0.30,
    "left_shoulder_pitch_joint":  0.90,   # arms forward toward handle
    "right_shoulder_pitch_joint": 0.90,
    "left_elbow_pitch_joint":     1.00,   # elbows bent to grip handle
    "right_elbow_pitch_joint":    1.00,
    "torso_joint":               -0.30,   # torso reclined back
}


# ── TARGET POSE (stable wakeboard ride, what the policy should learn) ──────
# Real wakeboarding riding stance:
#   - Slight crouch (knees never locked)
#   - Arms nearly straight (elbows ~5-10° from full extension)
#   - Handle at hip height
#   - Slight backward lean on torso (~15°)
#   - Feet balanced on board, board slightly nose-up

TARGET_ROOT_Z = 0.85
TARGET_ROOT_ROT = (0.7071, 0.0, 0.0, 0.7071)   # same sideways orientation

TARGET_JOINTS = {
    "left_hip_pitch_joint":      -0.15,   # slight crouch (not locked straight)
    "right_hip_pitch_joint":     -0.15,
    "left_knee_joint":            0.25,   # soft knee bend
    "right_knee_joint":           0.25,
    "left_ankle_pitch_joint":     0.10,   # slightly flat, heels down
    "right_ankle_pitch_joint":    0.10,
    "left_shoulder_pitch_joint":  0.40,   # arms slightly forward and up
    "right_shoulder_pitch_joint": 0.40,
    "left_elbow_pitch_joint":     0.10,   # nearly straight (crucial — rule #3)
    "right_elbow_pitch_joint":    0.10,
    "torso_joint":               -0.25,   # slight backward lean (~14°)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80,
                   help="sim steps to hold each pose before screenshot")
    p.add_argument("--num_envs", type=int, default=1)
    return p.parse_args()


def _set_pose(robot, root_z, root_rot_wxyz, joint_dict, device):
    import torch
    num_envs = robot.data.root_pos_w.shape[0]
    env_ids = torch.arange(num_envs, device=device)

    # root pose
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = root_z
    w, x, y, z = root_rot_wxyz
    root_state[:, 3] = w
    root_state[:, 4] = x
    root_state[:, 5] = y
    root_state[:, 6] = z
    root_state[:, 7:] = 0.0   # zero velocities
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    # joints
    jpos = robot.data.default_joint_pos.clone()
    jvel = torch.zeros_like(jpos)
    for i, name in enumerate(robot.joint_names):
        for pattern, val in joint_dict.items():
            if pattern in name:
                jpos[:, i] = val
                break
    robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
    return jpos


def _capture_screenshot(frame_dir, tag, step):
    """Trigger one replicator render and rename the output PNG."""
    try:
        import omni.replicator.core as rep, glob, shutil
        rep.orchestrator.step(delta_time=0.0)
        pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if pngs:
            dst = f"pose_{tag}.png"
            shutil.copy(pngs[-1], dst)
            print(f"[pose_viewer] Saved {dst}", flush=True)
    except Exception as e:
        print(f"[pose_viewer] Screenshot failed: {e}", flush=True)


def _print_comparison(cannonball_jpos, target_jpos, joint_names, device):
    import torch
    lines = []
    lines.append("\n" + "="*72)
    lines.append(f"{'JOINT NAME':<35} {'INITIAL (rad)':>14} {'TARGET (rad)':>14}")
    lines.append("-"*72)
    for i, name in enumerate(joint_names):
        iv = cannonball_jpos[0, i].item()
        tv = target_jpos[0, i].item()
        diff_marker = " ←" if abs(iv - tv) > 0.05 else ""
        lines.append(f"{name:<35} {iv:>14.3f} {tv:>14.3f}{diff_marker}")
    lines.append("="*72)
    lines.append(f"{'ROOT Z':35} {CANNONBALL_ROOT_Z:>14.3f} {TARGET_ROOT_Z:>14.3f}")
    lines.append(f"{'ROOT ROT (w,x,y,z)':35} {'(0.707,0,0,0.707)':>14} {'(0.707,0,0,0.707)':>14}")
    lines.append("="*72)
    lines.append("")
    lines.append("KEY DIFFERENCES (what the policy must learn):")
    lines.append("  hip_pitch:      -0.80 → -0.15  (unfold from crouch)")
    lines.append("  knee:            1.40 →  0.25  (extend legs, stay slightly bent)")
    lines.append("  elbow_pitch:     1.00 →  0.10  (straighten arms — rule #3)")
    lines.append("  shoulder_pitch:  0.90 →  0.40  (lower arms to hip height)")
    lines.append("  root Z:          0.50 →  0.85  (rise from water level to riding height)")
    lines.append("")
    lines.append("ORIENTATION: both poses face +Y (sideways to rope pull +X)")
    lines.append("HEAD NOTE:   head faces +Y — should face +X toward boat (deferred fix)")
    lines.append("="*72 + "\n")

    text = "\n".join(lines)
    print(text)
    with open("pose_comparison.txt", "w") as f:
        f.write(text)
    print("[pose_viewer] Saved pose_comparison.txt", flush=True)


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=True)
    app = launcher.app

    import torch
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg

    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = WakeboardStartEnv(env_cfg)
    device = str(env.device)
    robot = env.scene["robot"]

    # ── camera setup ───────────────────────────────────────────────────────
    frame_dir = "pose_frames"
    os.makedirs(frame_dir, exist_ok=True)
    try:
        import omni.replicator.core as rep
        # Side view: camera at Y=-5, looking at robot
        cam = rep.create.camera(position=(0.0, -5.0, 1.2), look_at=(0.0, 0.0, 0.85))
        rp = rep.create.render_product(cam, resolution=(1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=frame_dir, rgb=True)
        writer.attach([rp])
        # warmup
        for _ in range(5):
            rep.orchestrator.step(delta_time=0.016)
        print("[pose_viewer] Camera ready", flush=True)
    except Exception as e:
        print(f"[pose_viewer] Camera setup failed (screenshots disabled): {e}", flush=True)

    # ── initial reset ──────────────────────────────────────────────────────
    env.reset()

    # ── POSE 1: INITIAL (cannonball) ───────────────────────────────────────
    print("\n[pose_viewer] Setting INITIAL (cannonball) pose...", flush=True)
    cannon_jpos = _set_pose(robot, CANNONBALL_ROOT_Z, CANNONBALL_ROOT_ROT,
                            CANNONBALL_JOINTS, device)

    # hold pose with zero actions so it settles
    zero_action = torch.zeros(args.num_envs, env.action_manager.action_dim, device=device)
    for step in range(args.steps):
        env.step(zero_action)
        # re-apply pose each step to prevent physics from moving it
        _set_pose(robot, CANNONBALL_ROOT_Z, CANNONBALL_ROOT_ROT, CANNONBALL_JOINTS, device)

    print(f"[pose_viewer] Initial pose held for {args.steps} steps", flush=True)
    _capture_screenshot(frame_dir, "initial", args.steps)

    # print initial state diagnostics
    h = robot.data.root_pos_w[0, 2].item()
    q = robot.data.root_quat_w[0].tolist()
    print(f"[pose_viewer] INITIAL — pelvis Z={h:.3f}m  quat(wxyz)={[round(v,3) for v in q]}")
    print(f"[pose_viewer] Board weld count logged at startup above ^")
    bp = env._board_pitch[0].item() * 180.0 / math.pi
    print(f"[pose_viewer] Board pitch = {bp:.1f}° (expect close to 0° flat at spawn)")

    # ── POSE 2: TARGET (riding stance) ────────────────────────────────────
    print("\n[pose_viewer] Setting TARGET (riding) pose...", flush=True)
    target_jpos = _set_pose(robot, TARGET_ROOT_Z, TARGET_ROOT_ROT,
                            TARGET_JOINTS, device)

    for step in range(args.steps):
        env.step(zero_action)
        _set_pose(robot, TARGET_ROOT_Z, TARGET_ROOT_ROT, TARGET_JOINTS, device)

    print(f"[pose_viewer] Target pose held for {args.steps} steps", flush=True)
    _capture_screenshot(frame_dir, "target", args.steps)

    h = robot.data.root_pos_w[0, 2].item()
    bp = env._board_pitch[0].item() * 180.0 / math.pi
    print(f"[pose_viewer] TARGET — pelvis Z={h:.3f}m  board pitch={bp:.1f}°")

    # ── COMPARISON TABLE ───────────────────────────────────────────────────
    _print_comparison(cannon_jpos, target_jpos, robot.joint_names, device)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
