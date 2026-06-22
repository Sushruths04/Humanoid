"""
3D pose visualization — G1 robot on wakeboard, initial vs target.
Shows front / side / top views for both poses.
No Isaac Sim needed — runs locally with matplotlib.

Run:
    python visualize_poses_3d.py
Output:
    pose_3d_initial.png
    pose_3d_target.png
    pose_3d_comparison.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DEG = math.pi / 180.0

# ── colours ────────────────────────────────────────────────────────────────
COL_BONE   = "#2C3E50"
COL_JOINT  = "#E74C3C"
COL_BOARD  = "#D4A017"
COL_ROPE   = "#27AE60"
COL_COM    = "#8E44AD"
COL_BG     = "#F0F4F8"
COL_HEAD   = "#95A5A6"

# ── Forward kinematics for G1 in cannonball / target pose ─────────────────
# Coordinate system (matches sim):
#   +X = direction of travel (rope pulls this way)
#   +Y = direction robot faces (rider faces sideways)
#   +Z = up
#
# After 90° yaw rotation, G1's local sagittal plane = world Y-Z plane.
# So hip_pitch bends the knees in Y direction, not X.

def build_skeleton(hip_pitch, knee, ankle_pitch, shoulder_pitch, elbow_pitch,
                   torso_lean, pelvis_z,
                   hip_width=0.10,     # half hip width in X direction
                   thigh_len=0.28, shank_len=0.28, foot_len=0.09,
                   upper_arm=0.25, forearm=0.22):
    """
    Returns dict of joint world positions (x, y, z).
    Robot faces +Y. Legs bend in Y-Z plane (sagittal of rotated robot).
    Hip_pitch negative = knees come toward +Y (forward in robot frame).
    Torso_lean = backward lean in X direction (toward -X, opposing rope +X pull).
    """
    joints = {}

    # Pelvis (root)
    pelvis = np.array([0.0, 0.0, pelvis_z])
    joints["pelvis"] = pelvis

    # Torso top (~0.28m above pelvis, leaning in -X by torso_lean angle)
    torso_offset = np.array([
        -math.sin(torso_lean) * 0.28,   # backward lean in -X
         0.0,
         math.cos(torso_lean) * 0.28,
    ])
    torso = pelvis + torso_offset
    joints["torso"] = torso

    # Head (0.15m above torso top)
    head = torso + np.array([-math.sin(torso_lean)*0.12, 0.0, math.cos(torso_lean)*0.12])
    joints["head"] = head

    # LEFT foot (hip at +X side)
    for side, sx in [("L", +hip_width), ("R", -hip_width)]:
        hip = pelvis + np.array([sx, 0.0, 0.0])
        joints[f"hip_{side}"] = hip

        # Thigh: hip_pitch bends in robot's sagittal (Y-Z plane)
        # hip_pitch < 0 = forward flex → knee goes toward +Y
        thigh_dir = np.array([
            0.0,
            math.sin(-hip_pitch),       # negative hip_pitch → +Y component
            -math.cos(-hip_pitch),       # downward component
        ])
        knee_pos = hip + thigh_dir * thigh_len
        joints[f"knee_{side}"] = knee_pos

        # Shank: knee flexion further bends
        # Combined angle from vertical = hip_pitch + knee (both in same direction)
        combined = -hip_pitch + knee
        shank_dir = np.array([
            0.0,
            -math.sin(combined),        # goes back toward -Y at high knee flex
             -math.cos(combined),
        ])
        ankle_pos = knee_pos + shank_dir * shank_len
        joints[f"ankle_{side}"] = ankle_pos

        # Foot (mostly horizontal, slight pitch)
        foot_dir = np.array([0.0, math.sin(ankle_pitch), -math.cos(ankle_pitch)])
        toe_pos = ankle_pos + foot_dir * foot_len
        joints[f"toe_{side}"] = toe_pos

    # Shoulders (at torso top, offset in X)
    for side, sx in [("L", +0.18), ("R", -0.18)]:
        shoulder = torso + np.array([sx * math.cos(torso_lean), 0.0, sx * math.sin(torso_lean)])
        joints[f"shoulder_{side}"] = shoulder

        # Upper arm: shoulder_pitch moves arm in robot's sagittal plane (+Y for forward)
        # torso_lean tilts the whole frame so we need world coords
        ua_dir = np.array([
            -math.sin(torso_lean),        # tilted with torso
             math.sin(shoulder_pitch),
             math.cos(torso_lean) * (-1) + math.cos(shoulder_pitch) * 0,
        ])
        # simplified: upper arm hangs down from shoulder, pitched forward
        ua_dir = np.array([
            -math.sin(torso_lean) * 0.3,
             math.sin(shoulder_pitch),
            -math.cos(shoulder_pitch) * 0.5,
        ])
        ua_dir = ua_dir / (np.linalg.norm(ua_dir) + 1e-8)
        elbow_pos = shoulder + ua_dir * upper_arm
        joints[f"elbow_{side}"] = elbow_pos

        # Forearm: elbow_pitch bends further
        fa_dir_base = ua_dir.copy()
        # rotate fa around X axis by elbow_pitch
        bend = np.array([0.0, math.sin(elbow_pitch), -math.cos(elbow_pitch)])
        fa_dir = (ua_dir + 0.6 * bend)
        fa_dir = fa_dir / (np.linalg.norm(fa_dir) + 1e-8)
        hand_pos = elbow_pos + fa_dir * forearm
        joints[f"hand_{side}"] = hand_pos

    return joints


def board_verts(board_len=1.4, board_w=0.4, board_t=0.04):
    """Board box vertices. Long axis along Y (perpendicular to travel)."""
    # board center at origin Z=0.02, corners:
    hw = board_w / 2    # half width (X direction)  = 0.20
    hl = board_len / 2  # half length (Y direction) = 0.70
    t  = board_t
    faces = [
        # top face
        [[-hw,-hl,t],[ hw,-hl,t],[ hw, hl,t],[-hw, hl,t]],
        # bottom
        [[-hw,-hl,0],[ hw,-hl,0],[ hw, hl,0],[-hw, hl,0]],
        # front (+Y side)
        [[-hw, hl,0],[ hw, hl,0],[ hw, hl,t],[-hw, hl,t]],
        # back (-Y side)
        [[-hw,-hl,0],[ hw,-hl,0],[ hw,-hl,t],[-hw,-hl,t]],
        # left (-X)
        [[-hw,-hl,0],[-hw, hl,0],[-hw, hl,t],[-hw,-hl,t]],
        # right (+X)
        [[ hw,-hl,0],[ hw, hl,0],[ hw, hl,t],[ hw,-hl,t]],
    ]
    return [np.array(f) for f in faces]


def draw_skeleton_3d(ax, joints, color=COL_BONE, alpha=1.0, lw=3):
    """Draw bones and joints on a 3D axis."""
    # Bones to draw: (joint_a, joint_b)
    bones = [
        ("pelvis","torso"), ("torso","head"),
        ("hip_L","knee_L"), ("knee_L","ankle_L"), ("ankle_L","toe_L"),
        ("hip_R","knee_R"), ("knee_R","ankle_R"), ("ankle_R","toe_R"),
        ("pelvis","hip_L"), ("pelvis","hip_R"),
        ("torso","shoulder_L"), ("torso","shoulder_R"),
        ("shoulder_L","elbow_L"), ("elbow_L","hand_L"),
        ("shoulder_R","elbow_R"), ("elbow_R","hand_R"),
        ("hand_L","hand_R"),  # handle bar
    ]
    for a, b in bones:
        if a in joints and b in joints:
            pa, pb = joints[a], joints[b]
            ax.plot([pa[0],pb[0]], [pa[1],pb[1]], [pa[2],pb[2]],
                    color=color, lw=lw, alpha=alpha, solid_capstyle="round")

    # Joints as dots
    for name, pos in joints.items():
        if name == "head":
            continue
        ax.scatter([pos[0]], [pos[1]], [pos[2]],
                   color=COL_JOINT, s=40, zorder=5, alpha=alpha)

    # Head as sphere (approximate with scatter)
    h = joints["head"]
    ax.scatter([h[0]], [h[1]], [h[2]], color=COL_HEAD, s=600, alpha=0.5*alpha)

    # CoM (weighted average: 40% torso, 30% pelvis, 15% each knee)
    com = (0.4*joints["torso"] + 0.3*joints["pelvis"] +
           0.15*joints["knee_L"] + 0.15*joints["knee_R"])
    ax.scatter([com[0]], [com[1]], [com[2]],
               color=COL_COM, s=150, marker="*", zorder=6, alpha=alpha)
    return com


def draw_board_3d(ax, alpha=0.7):
    faces = board_verts()
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=COL_BOARD,
                            edgecolor="#8B6914", linewidth=0.8)
    ax.add_collection3d(poly)


def draw_rope_3d(ax, hand_mid, length=0.5, angle_deg=6):
    """Arrow from hand midpoint in +X direction (rope pull)."""
    angle = angle_deg * DEG
    tip = hand_mid + np.array([length, 0.0, length * math.tan(angle)])
    ax.quiver(hand_mid[0], hand_mid[1], hand_mid[2],
              tip[0]-hand_mid[0], tip[1]-hand_mid[1], tip[2]-hand_mid[2],
              color=COL_ROPE, linewidth=2.5, arrow_length_ratio=0.25)
    ax.text(tip[0]+0.05, tip[1], tip[2], "ROPE\n(+X, 600N)", color=COL_ROPE,
            fontsize=7.5, ha="left")


def make_pose_figure(title, joints, figsize=(16, 6)):
    """Single pose shown from 3 angles."""
    fig = plt.figure(figsize=figsize, facecolor=COL_BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)

    views = [
        ("Front View\n(looking in +Y direction: see X-Z plane)", -90, 0),
        ("Side View\n(looking in -X direction: see Y-Z plane)",   0, 0),
        ("Top View\n(looking down +Z: see X-Y plane)",           -90, 90),
    ]

    for idx, (view_title, azim, elev) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d", facecolor=COL_BG)
        ax.set_facecolor(COL_BG)

        draw_board_3d(ax)
        com = draw_skeleton_3d(ax, joints)

        # rope from midpoint of hands
        hand_mid = (joints["hand_L"] + joints["hand_R"]) / 2
        draw_rope_3d(ax, hand_mid)

        # ground plane
        xx, yy = np.meshgrid([-0.5, 0.5], [-0.9, 0.9])
        ax.plot_surface(xx, yy, np.zeros_like(xx),
                        alpha=0.15, color="#AED6F1")

        # axis labels with directions
        ax.set_xlabel("X →boat", fontsize=7, labelpad=0)
        ax.set_ylabel("Y →rider faces", fontsize=7, labelpad=0)
        ax.set_zlabel("Z ↑up", fontsize=7, labelpad=0)
        ax.set_xlim(-0.5, 0.7)
        ax.set_ylim(-0.9, 0.9)
        ax.set_zlim(-0.05, 1.4)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_title, fontsize=9, pad=4)
        ax.tick_params(labelsize=6)

        # annotations on side view
        if idx == 1:
            # pelvis height line
            px, py, pz = joints["pelvis"]
            ax.plot([px, px], [py, py], [0, pz],
                    color="#777", lw=1, linestyle=":")
            ax.text(px+0.08, py, pz/2,
                    f"pelvis\n{pz:.2f}m", fontsize=7, color="#555")

    # legend
    handles = [
        mpatches.Patch(color=COL_BOARD, label="Wakeboard (long axis = Y)"),
        plt.Line2D([0],[0], color=COL_ROPE, lw=2, label="Rope pull (+X)"),
        plt.scatter([],[], color=COL_COM, marker="*", s=80, label="Centre of Mass"),
        plt.scatter([],[], color=COL_JOINT, s=40, label="Joint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout()
    return fig


# ── BUILD POSES ────────────────────────────────────────────────────────────

INITIAL_JOINTS = build_skeleton(
    hip_pitch      = -0.80,
    knee           =  1.40,
    ankle_pitch    =  0.30,
    shoulder_pitch =  0.90,
    elbow_pitch    =  1.00,
    torso_lean     =  0.08,   # slight lean (~5°)
    pelvis_z       =  0.50,
)

TARGET_JOINTS = build_skeleton(
    hip_pitch      = -0.15,
    knee           =  0.25,
    ankle_pitch    =  0.10,
    shoulder_pitch =  0.35,
    elbow_pitch    =  0.10,
    torso_lean     =  0.45,   # 26° backward lean against rope
    pelvis_z       =  0.85,
)

# ── INDIVIDUAL POSE FIGURES ─────────────────────────────────────────────────
fig_init = make_pose_figure(
    "INITIAL POSE — Cannonball Deep-Water Start\n"
    "(pelvis 0.50m | deep crouch | arms bent | board flat)",
    INITIAL_JOINTS,
)
fig_init.savefig("pose_3d_initial.png", dpi=150, bbox_inches="tight", facecolor=COL_BG)
print("Saved: pose_3d_initial.png")
plt.close(fig_init)

fig_tgt = make_pose_figure(
    "TARGET POSE — Stable Riding at ~25 km/h\n"
    "(pelvis 0.85m | slight crouch | arms STRAIGHT | lean back 26° against rope)",
    TARGET_JOINTS,
)
fig_tgt.savefig("pose_3d_target.png", dpi=150, bbox_inches="tight", facecolor=COL_BG)
print("Saved: pose_3d_target.png")
plt.close(fig_tgt)

# ── COMPARISON FIGURE (both poses, side view only) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                          subplot_kw={"projection": "3d"},
                          facecolor=COL_BG)
fig.suptitle("Wakeboard Start — Initial vs Target Pose  (3D Side View)\n"
             "Rope pulls → +X   |   Rider faces → +Y   |   Board long axis along Y",
             fontsize=12, fontweight="bold")

for ax, joints, title, col in [
    (axes[0], INITIAL_JOINTS,
     "INITIAL (Cannonball)\npelvis=0.50m  elbow=1.00rad  torso lean=5°", COL_BONE),
    (axes[1], TARGET_JOINTS,
     "TARGET (Riding ~25 km/h)\npelvis=0.85m  elbow=0.10rad  torso lean=26°", "#1A5276"),
]:
    ax.set_facecolor(COL_BG)
    draw_board_3d(ax)
    com = draw_skeleton_3d(ax, joints, color=col)

    hand_mid = (joints["hand_L"] + joints["hand_R"]) / 2
    draw_rope_3d(ax, hand_mid)

    # ground
    xx, yy = np.meshgrid([-0.5, 0.5], [-0.9, 0.9])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color="#AED6F1")

    # pelvis height annotation
    px, py, pz = joints["pelvis"]
    ax.plot([px]*2, [py]*2, [0, pz], color="#AAA", lw=1, linestyle=":")
    ax.text(px+0.08, py, pz/2, f"{pz:.2f}m", fontsize=8, color="#555")

    # CoM annotation
    ax.text(com[0]-0.22, com[1], com[2], "CoM", fontsize=7,
            color=COL_COM, fontweight="bold")

    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y (rider faces)", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.set_xlim(-0.5, 0.7)
    ax.set_ylim(-0.9, 0.9)
    ax.set_zlim(-0.05, 1.45)
    ax.view_init(elev=8, azim=-60)   # slight 3D angle showing all axes
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(labelsize=6)

# key differences box
diff_text = (
    "What the policy learns (initial → target):\n"
    "  hip pitch:  -0.80 → -0.15 rad     knee:  1.40 → 0.25 rad\n"
    "  elbow:       1.00 →  0.10 rad   torso lean:  5° → 26° backward\n"
    "  pelvis Z:    0.50 →  0.85 m     (board stays on ground throughout)"
)
fig.text(0.5, 0.01, diff_text, ha="center", va="bottom", fontsize=9,
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92))

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("pose_3d_comparison.png", dpi=150, bbox_inches="tight", facecolor=COL_BG)
print("Saved: pose_3d_comparison.png")
plt.close(fig)

print("\nAll done:")
print("  pose_3d_initial.png     — cannonball (3 views)")
print("  pose_3d_target.png      — riding stance (3 views)")
print("  pose_3d_comparison.png  — side-by-side 3D comparison")
