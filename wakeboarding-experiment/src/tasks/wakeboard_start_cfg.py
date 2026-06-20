"""Wakeboard-start environment (PLAN.md §3-§4).

Manager-based Isaac Lab RL env: G1 (arms actuated) + welded board + rope pull, starting
from a crouched 'cannonball' pose, learning the deep-water start under a 30 km/h tow.

CPU-import-guarded. Everything Isaac-Lab-specific is inside the try-block; the `else`
branch gives a placeholder so the file imports for syntax checks without Isaac Sim.

VERIFY ON GPU (all marked inline): G1 link/joint names, projected_gravity_b, external-force
API (set_external_force_and_torque), fixed-joint foot binding, manager wiring versions.
"""
from __future__ import annotations

import math

import torch

from ..board import BoardParams, make_board_cfg, FOOT_BODY_NAMES
from ..rope_model import RopeModel, kmh_to_ms
from ..rewards import wakeboard_rewards as R

DEG = math.pi / 180.0

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    # reuse stock locomotion mdp helpers where possible
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

    ISAACLAB_AVAILABLE = True
except Exception:
    ISAACLAB_AVAILABLE = False
    def configclass(cls):  # type: ignore
        return cls


T_SUCCESS = 6.0  # seconds; success window (PLAN §2)

# --- G1 joint/link names, resolved from IsaacLab unitree.py G1_CFG (g1.usd, 23-DoF) ---
G1_ELBOW_JOINTS = ["left_elbow_pitch_joint", "right_elbow_pitch_joint"]
G1_KNEE_JOINTS = ["left_knee_joint", "right_knee_joint"]
G1_FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_TORSO_LINK = ["torso_link"]                       # VERIFY: torso_link vs pelvis on g1.usd
G1_HAND_LINKS = ["left_palm_link", "right_palm_link"]  # verified against live g1.usd body_names


# ============================================================ observations
def _board_pitch_obs(env):
    return env._board_pitch.unsqueeze(-1)


def _rope_force_obs(env):
    return env._rope_force


def _handle_rel_obs(env):
    return env._handle_pos - env._robot_root_pos


def _v_pull_obs(env):
    return env.rope.v_pull.unsqueeze(-1)


def _phase_obs(env):
    # episode_length_buf doesn't exist yet during the manager dim-resolution pass.
    elb = getattr(env, "episode_length_buf", None)
    if elb is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return torch.clamp(elb.float() * env.step_dt / T_SUCCESS, 0, 1).unsqueeze(-1)


if ISAACLAB_AVAILABLE:

    # -------------------------------------------------- scene
    @configclass
    class WakeboardSceneCfg(InteractiveSceneCfg):
        # ground = sand-like frictional plane
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.4, dynamic_friction=0.4, restitution=0.0
                )
            ),
        )
        # robot + board injected in __post_init__ (need params)
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
        )

    # -------------------------------------------------- observations
    @configclass
    class PolicyObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=loco_mdp.joint_pos_rel)          # VERIFY mdp names
        joint_vel = ObsTerm(func=loco_mdp.joint_vel_rel)
        base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=loco_mdp.projected_gravity)
        last_action = ObsTerm(func=loco_mdp.last_action)
        board_pitch = ObsTerm(func=_board_pitch_obs)
        rope_force = ObsTerm(func=_rope_force_obs)
        handle_rel = ObsTerm(func=_handle_rel_obs)
        v_pull = ObsTerm(func=_v_pull_obs)
        phase = ObsTerm(func=_phase_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyObsCfg = PolicyObsCfg()

    # -------------------------------------------------- actions
    @configclass
    class ActionsCfg:
        # joint-position control over ALL G1 joints (arms actuated, per task intent).
        # Mirrors the stock G1 velocity config (scale 0.5, default offset).
        joint_pos = loco_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

    # -------------------------------------------------- rewards (weights set from YAML)
    @configclass
    class RewardsCfg:
        pelvis_height = RewTerm(func=R.pelvis_height, weight=2.0)
        height_progress = RewTerm(func=R.height_progress, weight=1.5)
        uprightness = RewTerm(func=R.uprightness, weight=2.0)
        survival = RewTerm(func=R.survival, weight=0.5)
        forward_glide = RewTerm(func=R.forward_glide, weight=1.0)
        success_bonus = RewTerm(func=R.success_bonus, weight=50.0)
        board_positive_angle = RewTerm(func=R.board_positive_angle, weight=1.5)
        arms_straight = RewTerm(func=R.arms_straight, weight=1.0)
        handle_at_hips = RewTerm(func=R.handle_at_hips, weight=0.8)
        lean_back_moderate = RewTerm(func=R.lean_back_moderate, weight=0.7)
        knee_bend_maintained = RewTerm(func=R.knee_bend_maintained, weight=0.8)
        pen_stand_too_fast = RewTerm(func=R.pen_stand_too_fast, weight=-1.0)
        pen_pull_against_rope = RewTerm(func=R.pen_pull_against_rope, weight=-1.0)
        pen_torque = RewTerm(func=R.pen_torque, weight=-1e-4)
        pen_action_rate = RewTerm(func=R.pen_action_rate, weight=-0.01)
        pen_action_accel = RewTerm(func=R.pen_action_accel, weight=-1e-3)
        pen_dof_pos_limits = RewTerm(func=R.pen_dof_pos_limits, weight=-5.0)

    # -------------------------------------------------- terminations (PLAN §4.3)
    def _board_out_of_range(env):
        return (env._board_pitch < -20 * DEG) | (env._board_pitch > 45 * DEG)

    def _fell_over(env):
        return R.uprightness(env) < 0.3

    @configclass
    class TerminationsCfg:
        timeout = DoneTerm(func=loco_mdp.time_out, time_out=True)
        board_range = DoneTerm(func=_board_out_of_range)
        fell = DoneTerm(func=_fell_over)
        # VERIFY: add illegal-contact term for torso/head/knee using ContactSensor.

    def _reset_to_cannonball(env, env_ids):
        """Reset to crouched 'cannonball' wakeboard-start pose.

        Deep hip/knee flex, torso reclined, arms forward -- the deep-water start.
        """
        robot = env.scene["robot"]

        # Start from default root state and apply env origins
        default_root_state = robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # Override z to crouched height (lower CG -- crouched, not standing at 0.74m)
        default_root_state[:, 2] = env.scene.env_origins[env_ids, 2] + 0.50
        # Zero velocities
        default_root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        # Set cannonball joint positions
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        joint_names = robot.joint_names
        for i, name in enumerate(joint_names):
            if "hip_pitch" in name:
                joint_pos[:, i] = -0.8       # deep hip flexion
            elif "knee" in name:
                joint_pos[:, i] = 1.4        # deep knee flexion
            elif "ankle_pitch" in name:
                joint_pos[:, i] = 0.3        # feet relatively flat
            elif name == "torso_joint":
                joint_pos[:, i] = -0.3       # torso reclined back
            elif "shoulder_pitch" in name:
                joint_pos[:, i] = 0.9        # arms forward toward handle
            elif "elbow_pitch" in name:
                joint_pos[:, i] = 1.0        # elbows bent

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    @configclass
    class EventsCfg:
        reset_pose = EventTerm(func=_reset_to_cannonball, mode="reset")

    # -------------------------------------------------- env cfg
    @configclass
    class WakeboardStartEnvCfg(ManagerBasedRLEnvCfg):
        scene: WakeboardSceneCfg = WakeboardSceneCfg(num_envs=4096, env_spacing=4.0)
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventsCfg = EventsCfg()
        board: BoardParams = BoardParams()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 8.0
            self.sim.dt = 1.0 / 200.0
            # pull the stock G1 articulation (arms ACTUATED — do not freeze)
            g1 = G1FlatEnvCfg().scene.robot       # VERIFY: reuse the G1 ArticulationCfg
            g1.prim_path = "{ENV_REGEX_NS}/Robot"
            self.scene.robot = g1
            self.scene.board = make_board_cfg(self.board)

    # -------------------------------------------------- custom env (rope + buffers)
    class WakeboardStartEnv(ManagerBasedRLEnv):
        """Adds the rope force application + all biomechanics buffers used by rewards."""

        cfg: WakeboardStartEnvCfg

        def __init__(self, cfg, **kwargs):
            # Observation terms are evaluated once during super().__init__() (the manager
            # dim-resolution pass), so the custom buffers + rope they read must exist FIRST.
            num_envs = cfg.scene.num_envs
            device = getattr(cfg.sim, "device", None) or "cuda:0"
            self._t_success = T_SUCCESS
            self.rope = RopeModel(num_envs, device, model="spring", v_pull_kmh=10.0)
            self._init_buffers(num_envs, device)
            super().__init__(cfg, **kwargs)
            self._resolve_g1_indices()
            self._bind_feet_to_board()

        def _resolve_g1_indices(self):
            """Cache joint/body indices once (names from IsaacLab G1_CFG)."""
            robot = self.scene["robot"]
            self._elbow_idx = robot.find_joints(G1_ELBOW_JOINTS)[0]
            self._knee_idx = robot.find_joints(G1_KNEE_JOINTS)[0]
            try:
                self._hand_body_ids = robot.find_bodies(G1_HAND_LINKS)[0]
            except Exception:
                # fallback: any palm/hand link
                self._hand_body_ids = robot.find_bodies([".*_palm_link"])[0]
            self._torso_body_id = robot.find_bodies(G1_TORSO_LINK)[0]

        def _bind_feet_to_board(self):
            """Create fixed joints from both ankle-roll links to the board in each env.

            The board is spawned with its center at the ankle-roll link height, so the
            joint frames are coincident at reset and PhysX does not snap the bodies.
            """
            from pxr import Gf, Sdf, UsdGeom, UsdPhysics
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            created = 0
            for env_id in range(self.num_envs):
                board_path = f"/World/envs/env_{env_id}/Board"
                board_prim = stage.GetPrimAtPath(board_path)
                board_xf = UsdGeom.Xformable(board_prim).ComputeLocalToWorldTransform(0)
                board_inv = board_xf.GetInverse()
                for foot_name in G1_FOOT_LINKS:
                    foot_path = f"/World/envs/env_{env_id}/Robot/{foot_name}"
                    foot_prim = stage.GetPrimAtPath(foot_path)
                    foot_xf = UsdGeom.Xformable(foot_prim).ComputeLocalToWorldTransform(0)
                    foot_world = foot_xf.ExtractTranslation()
                    board_local = board_inv.Transform(foot_world)
                    joint_path = f"{board_path}/{foot_name}_fixed_joint"
                    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(foot_path)])
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(board_path)])
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(board_local[0]), float(board_local[1]), float(board_local[2])))
                    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    created += 1
            print(f"[wakeboard] board fixed joints configured: count={created}", flush=True)

        def _init_buffers(self, num_envs=None, device=None):
            num_envs = self.num_envs if num_envs is None else num_envs
            device = self.device if device is None else device
            z = lambda *s: torch.zeros(*s, device=device)
            self._board_pitch = z(num_envs)
            self._board_lin_vel = z(num_envs, 3)
            self._elbow_flexion = z(num_envs)
            self._knee_flexion = z(num_envs)
            self._torso_back_lean = z(num_envs)
            self._handle_pos = z(num_envs, 3)
            self._hip_target_pos = z(num_envs, 3)
            self._robot_root_pos = z(num_envs, 3)
            self._rope_force = z(num_envs, 3)
            self._success_event = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._fall_event = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._stable_time = z(num_envs)

        # --- core hook: apply rope force before ManagerBasedRLEnv writes scene data ---
        def step(self, action):
            self._refresh_biomech_buffers()
            self.rope.step_anchor(self.step_dt)
            force = self.rope.compute_force(self._handle_pos, self._handle_lin_vel())
            self._rope_force = force
            self._apply_handle_force(force)
            return super().step(action)

        def _apply_handle_force(self, force):
            # Apply the world-frame rope force across both palm links. Isaac Lab expects
            # (num_envs, num_bodies, 3), matching len(body_ids).
            robot = self.scene["robot"]
            num_hands = len(self._hand_body_ids)
            if num_hands == 0:
                raise RuntimeError("No hand body ids resolved for rope force application")
            forces = (force.unsqueeze(1).expand(-1, num_hands, -1) / num_hands).contiguous()
            torques = torch.zeros_like(forces)
            try:
                robot.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                    body_ids=self._hand_body_ids,
                    is_global=True,
                )
                if not getattr(self, "_reported_rope_force_apply", False):
                    self._reported_rope_force_apply = True
                    print(
                        "[wakeboard] rope force configured: "
                        f"force_shape={tuple(forces.shape)} "
                        f"body_ids={self._hand_body_ids} "
                        f"mean_norm={force.norm(dim=-1).mean().item():.3f}",
                        flush=True,
                    )
            except Exception as exc:
                if not getattr(self, "_reported_rope_force_error", False):
                    self._reported_rope_force_error = True
                    print(
                        "[wakeboard] rope force application failed: "
                        f"force_shape={tuple(forces.shape)} "
                        f"body_ids={self._hand_body_ids} error={exc!r}",
                        flush=True,
                    )
                raise

        def _handle_lin_vel(self):
            return self._board_lin_vel  # approx; VERIFY: use hand body velocity

        def _refresh_biomech_buffers(self):
            robot = self.scene["robot"]
            board = self.scene["board"]
            d = robot.data
            self._robot_root_pos = d.root_pos_w
            # board pitch + linear velocity
            self._board_lin_vel = board.data.root_lin_vel_w
            self._board_pitch = _quat_pitch(board.data.root_quat_w)   # VERIFY quat order (wxyz)
            # joint-angle biomechanics via cached G1 indices
            self._elbow_flexion = d.joint_pos[:, self._elbow_idx].abs().mean(dim=1)
            self._knee_flexion = d.joint_pos[:, self._knee_idx].abs().mean(dim=1)
            # handle = midpoint of the two hand bodies; torso back-lean = torso pitch
            hand_pos = d.body_pos_w[:, self._hand_body_ids]           # (N, 2, 3)
            self._handle_pos = hand_pos.mean(dim=1)
            torso_quat = d.body_quat_w[:, self._torso_body_id[0]]
            self._torso_back_lean = _quat_pitch(torso_quat).abs()
            self._hip_target_pos = d.root_pos_w + torch.tensor(
                [0.15, 0.0, 0.0], device=self.device)                # handle target near hips
            self._update_success_and_fall()

        def _update_success_and_fall(self):
            up = R.uprightness(self)
            h = self.scene["robot"].data.root_pos_w[:, 2]
            stable = (h >= 0.55) & (up >= 0.85) & (self._board_pitch.abs() < 30 * DEG)
            self._stable_time = torch.where(stable, self._stable_time + self.step_dt,
                                            torch.zeros_like(self._stable_time))
            newly = (self._stable_time >= 1.5) & (~self._success_event)
            self._success_event = newly
            self._fall_event = up < 0.3

        def reset(self, *args, **kwargs):
            out = super().reset(*args, **kwargs)
            # place the rope anchor ahead of the handle, reset stable timer
            self._refresh_biomech_buffers()
            self.rope.reset(torch.arange(self.num_envs, device=self.device), self._handle_pos)
            self._stable_time.zero_()
            self._success_event.zero_()
            return out



        def _reset_idx(self, env_ids):
            """Override to reset rope anchor + episode buffers for auto-reset envs."""
            super()._reset_idx(env_ids)
            self._refresh_biomech_buffers()
            self.rope.reset(env_ids, self._handle_pos)
            self._stable_time[env_ids] = 0.0
            self._success_event[env_ids] = False

# ============================================================ small geom helpers
def _quat_pitch(quat):  # (N,4) wxyz -> pitch radians ; VERIFY quat order (isaaclab uses wxyz)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
