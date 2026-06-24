"""Record a rollout trace + offscreen video from a checkpoint.

Usage (inside container):
    python play.py --checkpoint runs/wakeboard_stage1/model_300.pt --v_pull_kmh 10 --episodes 3 --out results/rollout.mp4
"""
from __future__ import annotations
import argparse, os, json, sys, subprocess, tempfile
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--v_pull_kmh", type=float, default=10.0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--out", default="results/rollout.mp4")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--no-cameras", action="store_true", help="disable RTX cameras (trace only, works without RTX GPU)")
    p.add_argument("--ignore-terminations", action="store_true", help="run for full --steps regardless of episode resets")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from isaaclab.app import AppLauncher
    enable_cam = not args.no_cameras
    app_launcher = AppLauncher(headless=True, enable_cameras=enable_cam)
    app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from src.tasks.wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg, T_SUCCESS
    from src.rope_model import kmh_to_ms

    env_cfg = WakeboardStartEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 8.0
    env = WakeboardStartEnv(env_cfg)
    env.rope.set_v_pull(kmh_to_ms(args.v_pull_kmh))

    rl_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(rl_env, _min_cfg(), log_dir=None, device=str(env.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(env.device))

    # Setup offscreen camera via BasicWriter (proven to write frames)
    # Clear frame dir before rollout so only motion frames are captured
    import shutil, glob as _glob
    frame_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "wb_frames")
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    frames_ok = False
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom
        # Side-view camera — x=0 so we only need to update x each step to track robot
        cam = rep.create.camera(position=(0.0, -6.0, 1.5), look_at=(0.0, 0.0, 0.85))
        _cam_path = str(cam.GetPath())
        rp = rep.create.render_product(cam, resolution=(1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=frame_dir, rgb=True)
        writer.attach([rp])
        # Warmup — clears these frames right after
        for _ in range(5):
            rep.orchestrator.step(delta_time=0.016)
        # Clear all warmup frames so only rollout motion is captured
        for f in _glob.glob(os.path.join(frame_dir, "*.png")):
            os.remove(f)
        print(f"[play] BasicWriter ready, warmup frames cleared, rollout starting...", flush=True)
        frames_ok = True
    except Exception as e:
        print(f"[play] Camera setup failed: {e}", flush=True)
        frames_ok = False

    # Run rollout
    obs, _ = env.reset()
    # Clear AGAIN — env.reset() generates many static frames during physics settle
    if frames_ok:
        for f in _glob.glob(os.path.join(frame_dir, "*.png")):
            os.remove(f)
        print(f"[play] Post-reset frames cleared. Rollout starting NOW.", flush=True)

    collected = 0
    trace = {"pelvis_z": [], "board_pitch": [], "fell": [], "reward": [], "step": [],
             "pelvis_x": [], "pelvis_vx": [], "rope_force": [], "uprightness": [],
             "joint_pos": [], "root_quat": []}
    ep_steps = torch.zeros(env.num_envs, device=env.device)

    for step in range(args.steps):
        with torch.no_grad():
            act = policy(obs)
        result = rl_env.step(act)
        obs, rewards = result[0], result[1]
        dones = result[2] if len(result) > 2 else torch.zeros(1, dtype=torch.bool, device=env.device)

        ep_steps += 1

        # Trace data — env 0 only for rich data
        robot = env.scene["robot"]
        h = robot.data.root_pos_w[0, 2].item()
        px = robot.data.root_pos_w[0, 0].item()
        vx = robot.data.root_lin_vel_w[0, 0].item()
        quat = robot.data.root_quat_w[0].cpu().tolist()
        jpos = robot.data.joint_pos[0].cpu().tolist()
        bp = env._board_pitch[0].item() * 180.0 / 3.14159
        fell = env._fall_event[0].float().item()
        rf = env._rope_force[0].norm().item()
        grav_up = -robot.data.projected_gravity_b[0, 2].item()
        trace["pelvis_z"].append(h)
        trace["pelvis_x"].append(px)
        trace["pelvis_vx"].append(vx)
        trace["root_quat"].append(quat)
        trace["joint_pos"].append(jpos)
        trace["board_pitch"].append(bp)
        trace["fell"].append(fell)
        trace["rope_force"].append(rf)
        trace["uprightness"].append(grav_up)
        trace["reward"].append(rewards[0].item() if torch.is_tensor(rewards) else float(rewards))
        trace["step"].append(step)

        # Capture frame — move camera to follow robot then render
        if frames_ok and step % 2 == 0:
            try:
                import omni.replicator.core as rep
                import omni.usd
                from pxr import Gf, UsdGeom
                # Track robot x position so robot stays in frame
                stage = omni.usd.get_context().get_stage()
                cam_prim = stage.GetPrimAtPath(_cam_path)
                for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps():
                    if UsdGeom.XformOp.TypeTranslate == op.GetOpType():
                        cur = op.Get()
                        op.Set(Gf.Vec3d(px, float(cur[1]), float(cur[2])))
                        break
                rep.orchestrator.step(delta_time=0.0)
            except Exception as e:
                print(f"[play] render step {step} failed: {e}", flush=True)
                frames_ok = False

        # Check done
        done_idx = dones.nonzero(as_tuple=False).flatten()
        for i in done_idx.tolist():
            collected += 1
            ep_steps[i] = 0
            if not args.ignore_terminations and collected >= args.episodes:
                break
        if not args.ignore_terminations and collected >= args.episodes:
            break

    # Save trace
    trace_out = args.out.replace(".mp4", "_trace.json")
    with open(trace_out, "w") as f:
        json.dump(trace, f)
    print(f"[play] Trace saved: {trace_out} ({len(trace['pelvis_z'])} steps)")

    # Encode video — rename frames sequentially then ffmpeg
    pngs = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    print(f"[play] Frames captured: {len(pngs)} in {frame_dir}", flush=True)
    if pngs:
        # Rename to sequential frame_%06d.png for ffmpeg
        for idx, fname in enumerate(pngs):
            src = os.path.join(frame_dir, fname)
            dst = os.path.join(frame_dir, f"seq_{idx:06d}.png")
            os.rename(src, dst)
        cmd = ["ffmpeg", "-y", "-framerate", "30", "-i",
               os.path.join(frame_dir, "seq_%06d.png"),
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-vf", "scale=1280:720", args.out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[play] Video saved: {args.out} ({len(pngs)} frames, 30fps, 1280x720)", flush=True)
        else:
            print(f"[play] ffmpeg failed: {result.stderr[-500:]}", flush=True)
    else:
        print("[play] No frames captured — check camera setup log above", flush=True)

    # Print summary
    print(f"\n=== ROLLOUT SUMMARY (model: {args.checkpoint}) ===")
    print(f"Steps: {len(trace['pelvis_z'])}  Episodes completed: {collected}")
    if trace["pelvis_z"]:
        print(f"Pelvis Z:  start={trace['pelvis_z'][0]:.3f}  max={max(trace['pelvis_z']):.3f}  end={trace['pelvis_z'][-1]:.3f}")
    if trace["board_pitch"]:
        print(f"Board pitch: start={trace['board_pitch'][0]:.1f}  max={max(trace['board_pitch']):.1f}  min={min(trace['board_pitch']):.1f}")
    if trace["fell"]:
        print(f"Fell: {sum(trace['fell']):.0f}/{len(trace['fell'])} steps")

    env.close()
    app.close()


def _min_cfg():
    hidden = [256, 256]
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    runner = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24, max_iterations=1, save_interval=1,
        experiment_name="play",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=hidden, activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=hidden, activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=5e-4, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )
    rl_cfg = runner.to_dict()
    for grp in ("actor", "critic"):
        for dep in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            if isinstance(rl_cfg.get(grp), dict):
                rl_cfg[grp].pop(dep, None)
    return rl_cfg


if __name__ == "__main__":
    main()
