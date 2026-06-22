"""
Wakeboard pose visualization — NO Isaac Sim needed.
Creates side-view diagrams of:
  - INITIAL pose (cannonball deep-water start)
  - TARGET pose  (stable riding at ~25-30 km/h)

Shows board, robot stick figure, rope direction, key angles.

Run locally:
    python visualize_poses.py
Outputs:
    pose_diagram.png  (side-by-side comparison, high-res)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc

# ── colours ────────────────────────────────────────────────────────────────
C_BOARD   = "#D4A017"   # warm gold
C_BONE    = "#2C3E50"   # dark blue-grey for skeleton
C_JOINT   = "#E74C3C"   # red dots for joints
C_ROPE    = "#27AE60"   # green rope
C_COM     = "#8E44AD"   # purple = centre of mass
C_ANGLE   = "#E67E22"   # orange = angle arc
C_BG      = "#F0F4F8"   # light background

DEG = math.pi / 180.0

# ───────────────────────────────────────────────────────────────────────────
# 2-D stick figure helper (viewing from +Y axis → X-Z plane)
# NOTE: we view from the SIDE perpendicular to the rider's facing direction.
#   - The rope pulls the rider in +X (screen-right)
#   - The rider stands SIDEWAYS (facing into/out of page)
#   - We see them from the side, so "backward lean" = lean toward -X (screen-left)
# ───────────────────────────────────────────────────────────────────────────

class Pose2D:
    """Minimal 2-D chain: ankle → knee → hip → torso → shoulder → elbow → hand"""
    def __init__(self, name,
                 ankle_z,        # ankle height above ground
                 knee_fwd,       # knee x offset from ankle (positive = forward = +X = toward boat)
                 knee_z,         # knee height
                 hip_x, hip_z,   # hip position
                 torso_x, torso_z,  # torso / shoulder position
                 lean_deg,       # backward lean of torso (positive = lean into -X = heelside)
                 hand_x, hand_z, # rope handle position
                 board_left, board_right,  # board extent in X
                 board_z=0.04,
                 elbow_x=None, elbow_z=None):
        self.name = name
        self.ankle  = np.array([0.0,   ankle_z])
        self.knee   = np.array([knee_fwd, knee_z])
        self.hip    = np.array([hip_x,  hip_z])
        self.torso  = np.array([torso_x, torso_z])
        self.shoulder = np.array([torso_x, torso_z])
        self.hand   = np.array([hand_x, hand_z])
        self.board_left  = board_left
        self.board_right = board_right
        self.board_z     = board_z
        self.lean_deg    = lean_deg
        self.elbow = np.array([elbow_x, elbow_z]) if elbow_x is not None else None

        # approx CoM: weighted midpoint of main segments
        self.com = (0.3*self.hip + 0.4*self.torso + 0.15*self.knee + 0.15*self.ankle)

    def draw(self, ax, x_offset=0.0, color=C_BONE, label_offset=(0.05, 0.05)):
        def pt(p):
            return (p[0] + x_offset, p[1])

        # board
        bx = [self.board_left + x_offset, self.board_right + x_offset,
              self.board_right + x_offset, self.board_left + x_offset,
              self.board_left + x_offset]
        bz = [0, 0, self.board_z, self.board_z, 0]
        ax.fill(bx, bz, color=C_BOARD, alpha=0.85, zorder=2)
        ax.plot(bx, bz, color="#8B6914", linewidth=1.5, zorder=3)

        # skeleton chain
        joints = [self.ankle, self.knee, self.hip, self.torso]
        for a, b in zip(joints[:-1], joints[1:]):
            ax.plot([a[0]+x_offset, b[0]+x_offset], [a[1], b[1]],
                    color=color, linewidth=4, solid_capstyle="round", zorder=5)

        # arm: shoulder → elbow (if given) → hand
        arm_pts = [self.shoulder]
        if self.elbow is not None:
            arm_pts.append(self.elbow)
        arm_pts.append(self.hand)
        for a, b in zip(arm_pts[:-1], arm_pts[1:]):
            ax.plot([a[0]+x_offset, b[0]+x_offset], [a[1], b[1]],
                    color=color, linewidth=3, solid_capstyle="round", zorder=5,
                    linestyle="--" if color == C_BONE else "-")

        # head
        head_r = 0.08
        hx = self.torso[0] + x_offset
        hz = self.torso[1] + head_r + 0.02
        head = plt.Circle((hx, hz), head_r, color=color, alpha=0.4, zorder=6)
        ax.add_patch(head)

        # joints as dots
        for pt_arr in [self.ankle, self.knee, self.hip, self.torso, self.hand]:
            ax.scatter([pt_arr[0]+x_offset], [pt_arr[1]],
                       color=C_JOINT, s=60, zorder=7)

        # CoM
        ax.scatter([self.com[0]+x_offset], [self.com[1]],
                   color=C_COM, s=120, marker="*", zorder=8, label="CoM")

        # label
        ax.text(self.com[0]+x_offset + label_offset[0],
                self.com[1] + label_offset[1],
                self.name, fontsize=11, fontweight="bold", color=color, zorder=9)


# ── INITIAL POSE  (cannonball deep-water start) ────────────────────────────
# Viewing from side (+Y direction → X is horizontal, Z is vertical)
# Rope pulls in +X direction (screen right).
# Rider is sideways → knees fold in the plane we're looking at (X-Z).
# Deep crouch: hip_pitch=-0.8, knee=1.4
#
# Approximate joint positions (relative to ankle at origin):
#   Thigh length ~ 0.28 m, shank ~ 0.28 m
#   hip_pitch = -0.8 rad (flexion: thigh goes forward/up)
#   thigh:  dx=sin(0.8)*0.28=+0.20, dz=cos(0.8)*0.28=+0.19
#   knee: (0.20, 0.04+0.19) = (0.20, 0.23)
#   lower_leg angle from vertical = 0.8-1.4 = -0.6 rad (goes backward)
#   shank:  dx=-sin(0.6)*0.28=-0.16, dz=cos(0.6)*0.28=+0.23
#   hip: knee + shank = (0.20-0.16, 0.23+0.23) = (0.04, 0.46)
#   pelvis root ~ same as hip = (0.04, 0.50)
#   torso (top): slight backward tilt → (-0.08, 0.50+0.30) = (-0.08, 0.80) -- WAIT
#   At CANNONBALL_ROOT_Z=0.50, pelvis is at z=0.50
#   Torso is ~0.30m above pelvis → torso_z = 0.80, torso slightly reclined → x ~ -0.03
#   Arms forward for handle: shoulder at torso, elbow bent 1.0 rad
#   hand at approximately (0.35, 0.55)

INITIAL = Pose2D(
    name="INITIAL\n(Cannonball)",
    ankle_z=0.04,          # on top of board (board top = 0.04 m)
    knee_fwd=+0.20,        # knee forward (+X, toward boat)
    knee_z=0.23,
    hip_x=+0.04,           # hip slightly forward
    hip_z=0.50,            # cannonball pelvis height
    torso_x=-0.03,         # torso slightly reclined back
    torso_z=0.78,          # torso top
    lean_deg=5.0,          # slight backward lean (not much yet)
    hand_x=+0.35,          # arms extended forward toward handle
    hand_z=0.55,           # handle at mid-torso height (bent elbows)
    board_left=-0.20,      # board is 0.4m along X (width)
    board_right=+0.20,
    elbow_x=+0.28, elbow_z=0.68,
)

# ── TARGET POSE  (stable riding at ~25-30 km/h) ───────────────────────────
# Physics of the backward lean:
#   Rope tension ~600 N pulling handle in +X.
#   To balance, CoM must be shifted toward -X (heelside).
#   rider leans back: hip shifts toward -X, torso leans back.
#   Legs: slight crouch (knee ~0.25 rad, hip_pitch ~-0.15 rad)
#   Arms: nearly STRAIGHT (elbow ~0.10 rad = almost fully extended)
#   Handle at hip height (not raised above hips)
#   Backward lean: torso tilts ~ 20-30° toward -X

TARGET = Pose2D(
    name="TARGET\n(Riding ~25 km/h)",
    ankle_z=0.04,          # still on board
    knee_fwd=-0.05,        # knees very slightly back (heelside)
    knee_z=0.25,
    hip_x=-0.08,           # hip shifted backward (heelside, -X)
    hip_z=0.75,
    torso_x=-0.18,         # torso leaned back significantly (anti-rope)
    torso_z=1.00,          # standing taller
    lean_deg=22.0,         # ~22° backward lean to counter 600N pull
    hand_x=+0.40,          # arms reach FORWARD to handle (nearly straight)
    hand_z=0.80,           # handle at hip/lower-chest height
    board_left=-0.20,
    board_right=+0.20,
    elbow_x=+0.22, elbow_z=0.88,
)

# ── FIGURE ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor=C_BG)
fig.suptitle("Wakeboard Start — Initial vs Target Pose\n(Side View: Rope pulls → RIGHT  |  Rider leans ← LEFT to counter pull)",
             fontsize=14, fontweight="bold", y=0.97)

for ax, pose, xoff in [(axes[0], INITIAL, 0.0), (axes[1], TARGET, 0.0)]:
    ax.set_facecolor(C_BG)

    # ground / water surface
    ax.axhline(0, color="#7F8C8D", linewidth=1.5, linestyle="--", alpha=0.6, label="Ground/Water")
    ax.fill_between([-0.8, 1.2], -0.12, 0, color="#AED6F1", alpha=0.4)

    # draw the pose
    pose.draw(ax, x_offset=xoff)

    # ── rope arrow (from handle toward boat in +X direction) ──────────────
    hx, hz = pose.hand[0]+xoff, pose.hand[1]
    rope_end_x = hx + 0.55
    rope_angle_deg = 6.0   # slight upward angle (boat pylon elevated ~1m above water)
    rope_end_z = hz + 0.55 * math.tan(rope_angle_deg * DEG)
    ax.annotate("", xy=(rope_end_x, rope_end_z), xytext=(hx, hz),
                arrowprops=dict(arrowstyle="-|>", color=C_ROPE,
                                lw=2.5, mutation_scale=20))
    ax.text(rope_end_x + 0.04, rope_end_z,
            f"ROPE PULL\n(+X, 6° up)\n≈600 N",
            color=C_ROPE, fontsize=9, va="center")

    # ── backward lean angle arc ──────────────────────────────────────────
    if pose.lean_deg > 2:
        arc = Arc((pose.ankle[0]+xoff, pose.ankle[1]),
                  width=0.28, height=0.28,
                  angle=0, theta1=90-pose.lean_deg, theta2=90,
                  color=C_ANGLE, linewidth=2)
        ax.add_patch(arc)
        ax.text(pose.ankle[0]+xoff - 0.18, pose.ankle[1] + 0.16,
                f"{pose.lean_deg:.0f}°\nlean", color=C_ANGLE, fontsize=8.5)

    # ── key dimension annotations ─────────────────────────────────────────
    # pelvis height
    ax.annotate("", xy=(pose.hip[0]+xoff - 0.35, pose.hip[1]),
                xytext=(pose.hip[0]+xoff - 0.35, 0),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax.text(pose.hip[0]+xoff - 0.55, pose.hip[1]/2,
            f"pelvis\n{pose.hip[1]:.2f} m", color="#555", fontsize=8, ha="center")

    # arm straight note
    if pose.lean_deg > 10:
        ax.text(pose.hand[0]+xoff - 0.05, pose.hand[1] - 0.12,
                "Arms STRAIGHT\n(rule #3 — never\nbend elbows!)",
                color="#C0392B", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    else:
        ax.text(pose.hand[0]+xoff - 0.05, pose.hand[1] - 0.12,
                "Arms bent\n(grip handle\nfor start)",
                color="#7F8C8D", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # ── axes, titles ─────────────────────────────────────────────────────
    ax.set_xlim(-0.85, 1.15)
    ax.set_ylim(-0.18, 1.45)
    ax.set_aspect("equal")
    ax.set_xlabel("X  (→ = toward boat / direction of travel)", fontsize=10)
    ax.set_ylabel("Z  (↑ = up)", fontsize=10)
    ax.set_title(pose.name, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # ── legend items ─────────────────────────────────────────────────────
    ax.scatter([], [], color=C_JOINT, s=60, label="Joint")
    ax.scatter([], [], color=C_COM, s=120, marker="*", label="Centre of Mass")
    ax.plot([], [], color=C_ROPE, linewidth=2, label="Rope force")
    ax.fill([], [], color=C_BOARD, alpha=0.85, label="Wakeboard")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)


# ── shared annotation box ───────────────────────────────────────────────────
notes = (
    "WHAT THE POLICY MUST LEARN  (gap = initial → target)\n"
    "  • Hip pitch:   -0.80 → -0.15 rad  (unfold from cannonball)\n"
    "  • Knee:         1.40 →  0.25 rad  (extend legs, stay softly bent)\n"
    "  • Elbow:        1.00 →  0.10 rad  (straighten arms — MOST CRITICAL)\n"
    "  • Torso lean:    5°  →   22°      (lean back against 600 N rope pull)\n"
    "  • Pelvis Z:     0.50 →  0.85 m   (rise from water level to riding height)\n\n"
    "HEAD NOTE: After 90° yaw, G1 head faces +Y (sideways). Should face +X toward boat.\n"
    "           Deferred until vision / human-coaching phase."
)
fig.text(0.5, 0.01, notes, ha="center", va="bottom", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#BDC3C7", alpha=0.95),
         fontfamily="monospace")

plt.tight_layout(rect=[0, 0.14, 1, 0.96])
plt.savefig("pose_diagram.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
print("Saved: pose_diagram.png")
plt.close()


# ── PHASE 2 diagram ─────────────────────────────────────────────────────────
# Phase 2: after stable ride achieved, transition to natural wakeboarding stance
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8), facecolor=C_BG)
ax2.set_facecolor(C_BG)
fig2.suptitle("Phase 2 Goal — Natural Wakeboarding Stance (after successful start)",
              fontsize=13, fontweight="bold")

# ground
ax2.axhline(0, color="#7F8C8D", linewidth=1.5, linestyle="--", alpha=0.6)
ax2.fill_between([-0.8, 1.2], -0.12, 0, color="#AED6F1", alpha=0.4)

# Phase 2 pose: more upright, more natural, still sideways
PHASE2 = Pose2D(
    name="PHASE 2\n(Natural Riding)",
    ankle_z=0.04,
    knee_fwd=-0.03,
    knee_z=0.28,
    hip_x=-0.06,
    hip_z=0.80,
    torso_x=-0.12,
    torso_z=1.05,
    lean_deg=14.0,          # less lean at slower/stable speed
    hand_x=+0.38,
    hand_z=0.82,
    board_left=-0.20,
    board_right=+0.20,
    elbow_x=+0.20, elbow_z=0.90,
)
PHASE2.draw(ax2, x_offset=0.0,
            label_offset=(0.12, 0.0))

# rope
hx, hz = PHASE2.hand[0], PHASE2.hand[1]
ax2.annotate("", xy=(hx+0.55, hz+0.055), xytext=(hx, hz),
            arrowprops=dict(arrowstyle="-|>", color=C_ROPE, lw=2.5, mutation_scale=20))
ax2.text(hx+0.60, hz+0.06, "ROPE\n(continuous\npull at cruise speed)",
         color=C_ROPE, fontsize=9)

# transition arrow showing movement from target to phase2
ax2.annotate("transition from\nTarget pose here",
             xy=(-0.12, 1.00), xytext=(-0.60, 1.20),
             arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.5),
             fontsize=9, color="#8E44AD")

ax2.set_xlim(-0.85, 1.20)
ax2.set_ylim(-0.18, 1.45)
ax2.set_aspect("equal")
ax2.set_xlabel("X  (→ = direction of travel)", fontsize=10)
ax2.set_ylabel("Z  (↑ = up)", fontsize=10)
ax2.grid(True, alpha=0.3, linewidth=0.5)

phase2_notes = (
    "Phase 2 starts AFTER the Phase 1 success condition is met (stable ride ≥1.5 s).\n"
    "Goal: maintain a comfortable, natural riding stance while being towed at cruise speed.\n"
    "Differences from Phase 1 target:\n"
    "  • Slightly more upright (less aggressive backward lean)\n"
    "  • Softer knee bend (more relaxed, long-ride stance)\n"
    "  • Same arm position (straight, handle at hips)\n"
    "  • Board tracking flat (not nose-up any more — planing completed)"
)
ax2.text(0.5, -0.02, phase2_notes, transform=ax2.transAxes,
         ha="center", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
         fontfamily="monospace")

plt.tight_layout()
plt.savefig("pose_diagram_phase2.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
print("Saved: pose_diagram_phase2.png")
plt.close()

print("\nDone. Check:")
print("  pose_diagram.png       — initial vs target (side view)")
print("  pose_diagram_phase2.png — phase 2 natural riding stance")
