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

    # Setup offscreen camera via omni.replicator
    frame_dir = tempfile.mkdtemp(prefix="wb_frames_")
    frames_ok = False
    try:
        import omni.replicator.core as rep
        cam = rep.create.camera(position=(2.0, -3.5, 1.8), look_at=(0.0, 0.0, 0.65))
        rp = rep.create.render_product(cam, resolution=(640, 480))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([rp])
        rep.orchestrator.step(delta_time=0.0)  # warm up renderer
        print(f"[play] Offscreen camera attached to render product", flush=True)
        frames_ok = True
    except Exception as e:
        print(f"[play] Camera setup failed: {e}", flush=True)
        frames_ok = False

    # Run rollout
    obs, _ = env.reset()
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

        # Capture frame
        if frames_ok and step % 3 == 0:
            try:
                import omni.replicator.core as rep
                rep.orchestrator.step(delta_time=0.0)
                raw = rgb_annotator.get_data()
                if raw is not None:
                    # Isaac Sim 5.x returns {"data": ndarray, "info": {...}}
                    img_arr = raw["data"] if isinstance(raw, dict) else np.array(raw)
                    if img_arr is not None and img_arr.ndim == 3 and img_arr.shape[2] >= 3:
                        from PIL import Image
                        Image.fromarray(img_arr[:, :, :3]).save(
                            os.path.join(frame_dir, f"frame_{step:05d}.png"))
            except Exception:
                pass

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

    # Encode video if frames exist
    pngs = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if pngs:
        cmd = ["ffmpeg", "-y", "-framerate", "20", "-i",
               os.path.join(frame_dir, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out]
        subprocess.run(cmd, check=True)
        print(f"[play] Video saved: {args.out} ({len(pngs)} frames)")
    else:
        print("[play] No frames captured (offscreen rendering not available on this GPU)")

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
