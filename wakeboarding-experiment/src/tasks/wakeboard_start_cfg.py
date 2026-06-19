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
    return torch.clamp(env.episode_length_buf.float() * env.step_dt / T_SUCCESS, 0, 1).unsqueeze(-1)


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

    @configclass
    class EventsCfg:
        reset_pose = EventTerm(func=loco_mdp.reset_scene_to_default, mode="reset")
        # VERIFY: replace with a custom 'reset to cannonball crouch' event (see env.reset).

    # -------------------------------------------------- env cfg
    @configclass
    class WakeboardStartEnvCfg(ManagerBasedRLEnvCfg):
        scene: WakeboardSceneCfg = WakeboardSceneCfg(num_envs=4096, env_spacing=4.0)
        observations: ObservationsCfg = ObservationsCfg()
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
            super().__init__(cfg, **kwargs)
            self.rope = RopeModel(self.num_envs, self.device, model="spring", v_pull_kmh=10.0)
            self._t_success = T_SUCCESS
            self._init_buffers()

        def _init_buffers(self):
            z = lambda *s: torch.zeros(*s, device=self.device)
            self._board_pitch = z(self.num_envs)
            self._board_lin_vel = z(self.num_envs, 3)
            self._elbow_flexion = z(self.num_envs)
            self._knee_flexion = z(self.num_envs)
            self._torso_back_lean = z(self.num_envs)
            self._handle_pos = z(self.num_envs, 3)
            self._hip_target_pos = z(self.num_envs, 3)
            self._robot_root_pos = z(self.num_envs, 3)
            self._rope_force = z(self.num_envs, 3)
            self._success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._fall_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._stable_time = z(self.num_envs)

        # --- core hook: apply rope force + refresh buffers every physics decimation ---
        def _pre_physics_step(self, actions):
            super()._pre_physics_step(actions)
            self._refresh_biomech_buffers()
            self.rope.step_anchor(self.step_dt)
            force = self.rope.compute_force(self._handle_pos, self._handle_lin_vel())
            self._rope_force = force
            self._apply_handle_force(force)   # VERIFY external-force API below

        def _apply_handle_force(self, force):
            # VERIFY: apply at the hands/handle body via
            # robot.set_external_force_and_torque(forces, torques, body_ids=hand_ids)
            robot = self.scene["robot"]
            try:
                robot.set_external_force_and_torque(
                    force.unsqueeze(1), torch.zeros_like(force).unsqueeze(1),
                    body_ids=self._hand_body_ids,
                )
            except Exception:
                pass  # filled in on GPU once body ids are known

        def _handle_lin_vel(self):
            return self._board_lin_vel  # approx; VERIFY: use hand body velocity

        def _refresh_biomech_buffers(self):
            robot = self.scene["robot"]
            board = self.scene["board"]
            d = robot.data
            self._robot_root_pos = d.root_pos_w
            # board pitch from board orientation; board lin vel
            self._board_lin_vel = board.data.root_lin_vel_w           # VERIFY
            self._board_pitch = _quat_pitch(board.data.root_quat_w)   # VERIFY
            # joint-angle-derived biomechanics (need G1 joint index map) -- VERIFY indices
            self._elbow_flexion = _joint_angle(d, ("left_elbow_joint", "right_elbow_joint"))
            self._knee_flexion = _joint_angle(d, ("left_knee_joint", "right_knee_joint"))
            self._torso_back_lean = _torso_lean(d)
            self._handle_pos = _hand_midpoint(d)                      # VERIFY hand bodies
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


# ============================================================ small geom helpers
def _quat_pitch(quat):  # (N,4) wxyz -> pitch radians ; VERIFY quat order
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))


def _joint_angle(data, names):  # mean |angle| over named joints ; VERIFY name->index
    return torch.zeros(data.joint_pos.shape[0], device=data.joint_pos.device)  # FILL on GPU


def _torso_lean(data):
    return torch.zeros(data.root_pos_w.shape[0], device=data.root_pos_w.device)  # FILL on GPU


def _hand_midpoint(data):
    return data.root_pos_w + 0.0  # FILL: midpoint of the two hand bodies
