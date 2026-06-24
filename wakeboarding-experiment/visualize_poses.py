"""
Wakeboard pose visualization — NO Isaac Sim needed.
Creates diagrams of:
  - INITIAL pose (cannonball deep-water start)
  - TARGET pose  (stable riding at ~25-30 km/h)

View choice: looking from +X direction (from the BOAT side) = Y-Z plane.
After 90° yaw the robot faces +Y, knees bend in Y-Z — this view shows the
full depth of the crouch correctly. Rope pulls toward viewer (+X = into page),
shown as a circle with dot symbol.

Run locally:
    python visualize_poses.py
Outputs:
    pose_diagram.png        (side-by-side comparison, high-res)
    pose_diagram_phase2.png (phase 2 natural riding stance)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc

# ── colours ────────────────────────────────────────────────────────────────
C_BOARD  = "#D4A017"
C_BONE   = "#2C3E50"
C_JOINT  = "#E74C3C"
C_ROPE   = "#27AE60"
C_COM    = "#8E44AD"
C_ANGLE  = "#E67E22"
C_BG     = "#F0F4F8"

DEG = math.pi / 180.0

# ───────────────────────────────────────────────────────────────────────────
# VIEW: from +X direction → we see the Y-Z plane.
#   +Y = direction robot faces (horizontal in this view)
#   +Z = up (vertical in this view)
#   +X = into page (rope pulls toward viewer, shown as ⊙ symbol)
#
# With 90° yaw:
#   - board long axis is along Y  → shows as wide rectangle in this view
#   - knees bend forward in +Y (robot's sagittal) → visible here
#   - backward lean opposes rope by leaning in -Y (heelside)
# ───────────────────────────────────────────────────────────────────────────

def fk_leg(pelvis_z, hip_pitch, knee_flex, ankle_pitch,
           thigh=0.28, shank=0.28, foot=0.09):
    """
    Forward kinematics for one leg in the Y-Z plane.
    Robot faces +Y. hip_pitch < 0 = forward (knee goes +Y).
    Returns (knee_y, knee_z, ankle_y, ankle_z, toe_y, toe_z).
    """
    h = abs(hip_pitch)                         # magnitude of forward flex
    # thigh direction: forward (+Y) and down (-Z)
    thigh_y =  math.sin(h) * thigh
    thigh_z = -math.cos(h) * thigh
    knee_y = thigh_y
    knee_z = pelvis_z + thigh_z

    # shank: bends further back-and-down; combined rotation from downward -Z
    # At full crouch combined can exceed π/2 — shank swings backward (-Y)
    combined = h + knee_flex                   # total rotation from -Z
    if combined <= math.pi:
        shank_y = -math.sin(combined - h) * shank   # goes backward
        shank_z = -math.cos(combined - h) * shank   # goes down
        # Better: project in Y-Z from knee at angle (π - combined) from vertical
        # shank angle from downward vertical is combined, measured CCW from -Z toward +Y
        # so shank dir: +Y component = sin(combined), -Z component = cos(combined)
        # but already past horizontal when combined>π/2 so:
        # shank_y component from knee = -sin(π - combined) = -sin(combined... )
        # Let me just use the angle directly:
        s_y = math.sin(h) * shank - math.sin(combined) * shank   # simplified
        s_z = -math.cos(h) * shank - (math.cos(combined)) * shank
        # Actually compute properly:
        # shank direction from vertical: angle = combined from downward -Z toward +Y then past
        # direction vector: [sin(combined), -cos(combined)] but capped at going -Y past π/2
        dy = math.sin(combined)
        dz = -math.cos(combined)
        # shank goes from knee in this direction
        ankle_y = knee_y + dy * shank
        ankle_z = knee_z + dz * shank
    else:
        dy = math.sin(combined)
        dz = -math.cos(combined)
        ankle_y = knee_y + dy * shank
        ankle_z = knee_z + dz * shank

    # foot (mostly flat, slight pitch)
    toe_y = ankle_y + math.sin(ankle_pitch) * foot
    toe_z = ankle_z - math.cos(ankle_pitch) * foot
    return knee_y, knee_z, ankle_y, ankle_z, toe_y, toe_z


class Pose2D:
    """
    Stick figure viewed from +X direction (Y = horizontal, Z = vertical).
    Pelvis at (0, pelvis_z). Robot faces +Y.
    """
    def __init__(self, name, pelvis_z, hip_pitch, knee_flex, ankle_pitch,
                 shoulder_pitch, elbow_flex, torso_lean_y,
                 board_front, board_back, board_z=0.04,
                 torso_height=0.30, arm_upper=0.25, arm_forearm=0.22):
        self.name = name
        self.pelvis_z = pelvis_z

        # Leg FK
        ky, kz, ay, az, ty, tz = fk_leg(pelvis_z, hip_pitch, knee_flex, ankle_pitch)
        self.knee   = np.array([ky,  kz])
        self.ankle  = np.array([ay,  az])
        self.toe    = np.array([ty,  tz])
        self.pelvis = np.array([0.0, pelvis_z])

        # Torso: leans in +Y or -Y (backward lean = -Y to oppose rope +X pull)
        # torso_lean_y > 0 means lean toward -Y (heelside, opposing rope)
        torso_dy = -math.sin(torso_lean_y) * torso_height
        torso_dz =  math.cos(torso_lean_y) * torso_height
        self.torso = self.pelvis + np.array([torso_dy, torso_dz])

        # Head
        head_dy = -math.sin(torso_lean_y) * 0.15
        head_dz =  math.cos(torso_lean_y) * 0.15
        self.head = self.torso + np.array([head_dy, head_dz])

        # Arm: from torso forward (+Y) and slightly down
        ua_dy = math.cos(torso_lean_y) * math.sin(shoulder_pitch) * arm_upper
        ua_dz = -math.sin(shoulder_pitch) * arm_upper * 0.4
        self.elbow = self.torso + np.array([ua_dy, ua_dz])
        fa_dy = ua_dy + math.cos(torso_lean_y) * math.sin(elbow_flex) * arm_forearm * 0.5
        fa_dz = ua_dz - elbow_flex * arm_forearm * 0.15
        self.hand = self.elbow + np.array([fa_dy * 0.5, fa_dz])

        # CoM
        self.com = (0.35 * self.pelvis + 0.4 * self.torso +
                    0.15 * self.knee + 0.10 * self.ankle)

        self.board_front = board_front   # +Y extent
        self.board_back  = board_back    # -Y extent
        self.board_z     = board_z
        self.lean_deg    = math.degrees(torso_lean_y)

    def draw(self, ax, y_offset=0.0, color=C_BONE):
        def y(pt): return pt[0] + y_offset
        def z(pt): return pt[1]

        # Board (viewed from +X: shows full Y-Z cross-section)
        by = [self.board_back+y_offset, self.board_front+y_offset,
              self.board_front+y_offset, self.board_back+y_offset,
              self.board_back+y_offset]
        bz = [0, 0, self.board_z, self.board_z, 0]
        ax.fill(by, bz, color=C_BOARD, alpha=0.85, zorder=2)
        ax.plot(by, bz, color="#8B6914", linewidth=1.5, zorder=3)

        # Leg chain (one leg, mirrored for visual clarity)
        for pts in [
            [self.ankle, self.knee, self.pelvis],
        ]:
            for a, b in zip(pts[:-1], pts[1:]):
                ax.plot([y(a), y(b)], [z(a), z(b)],
                        color=color, linewidth=4,
                        solid_capstyle="round", zorder=5)
        # foot
        ax.plot([y(self.ankle), y(self.toe)], [z(self.ankle), z(self.toe)],
                color=color, linewidth=3, solid_capstyle="round", zorder=5)

        # Torso
        ax.plot([y(self.pelvis), y(self.torso)], [z(self.pelvis), z(self.torso)],
                color=color, linewidth=4, solid_capstyle="round", zorder=5)

        # Arm: torso → elbow → hand
        for a, b in [(self.torso, self.elbow), (self.elbow, self.hand)]:
            ax.plot([y(a), y(b)], [z(a), z(b)],
                    color=color, linewidth=3,
                    solid_capstyle="round", zorder=5, linestyle="--")

        # Head
        head_r = 0.08
        hcirc = plt.Circle((y(self.head), z(self.head)),
                            head_r, color=color, alpha=0.4, zorder=6)
        ax.add_patch(hcirc)

        # Joint dots
        for pt in [self.ankle, self.knee, self.pelvis, self.torso, self.hand]:
            ax.scatter([y(pt)], [z(pt)], color=C_JOINT, s=60, zorder=7)

        # CoM
        ax.scatter([y(self.com)], [z(self.com)],
                   color=C_COM, s=140, marker="*", zorder=8)

        # Label
        ax.text(y(self.com) + 0.05, z(self.com) + 0.06,
                self.name, fontsize=11, fontweight="bold",
                color=color, zorder=9)


# ── INITIAL POSE (cannonball) ──────────────────────────────────────────────
# hip_pitch=-0.8, knee=1.4 → ankles come back toward pelvis
# Board spans Y: -0.70 to +0.70 (1.4 m long axis = Y direction)
INITIAL = Pose2D(
    name="INITIAL\n(Cannonball)",
    pelvis_z=0.50,
    hip_pitch=-0.80,    # deep hip flexion → knees come forward (+Y)
    knee_flex=1.40,     # deep knee flex → shank swings back
    ankle_pitch=0.30,
    shoulder_pitch=0.90,
    elbow_flex=1.00,    # arms bent, gripping handle
    torso_lean_y=0.08,  # slight backward lean (5°) at start
    board_front=+0.70,  # board full 1.4m along Y (visible in this view)
    board_back=-0.70,
)

# ── TARGET POSE (stable riding) ────────────────────────────────────────────
# Legs less flexed, torso leans back 22° to oppose rope pull
TARGET = Pose2D(
    name="TARGET\n(Riding ~25 km/h)",
    pelvis_z=0.85,
    hip_pitch=-0.15,
    knee_flex=0.25,
    ankle_pitch=0.10,
    shoulder_pitch=0.35,
    elbow_flex=0.10,    # arms nearly straight
    torso_lean_y=0.38,  # 22° backward lean to oppose 600N rope
    board_front=+0.70,
    board_back=-0.70,
)

# ── FIGURE ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor=C_BG)
fig.suptitle(
    "Wakeboard Start — Initial vs Target Pose\n"
    "View from BOAT side (+X direction) → shows Y-Z plane\n"
    "Board long axis = Y (1.4 m visible)  |  Rope pulls into page (⊙ = toward viewer)",
    fontsize=13, fontweight="bold", y=0.98)

for ax, pose in [(axes[0], INITIAL), (axes[1], TARGET)]:
    ax.set_facecolor(C_BG)

    # ground / water
    ax.axhline(0, color="#7F8C8D", linewidth=1.5,
               linestyle="--", alpha=0.6, label="Ground/Water")
    ax.fill_between([-0.85, 0.85], -0.12, 0, color="#AED6F1", alpha=0.4)

    pose.draw(ax)

    # Rope symbol: ⊙ = coming toward viewer (+X into page)
    hy, hz = pose.hand[0], pose.hand[1]
    ax.scatter([hy], [hz], s=400, facecolors="white",
               edgecolors=C_ROPE, linewidths=2.5, zorder=10)
    ax.scatter([hy], [hz], s=80, color=C_ROPE, zorder=11)
    ax.text(hy + 0.08, hz + 0.04,
            "⊙ ROPE\n(+X toward viewer\n≈600 N pull)",
            color=C_ROPE, fontsize=9, va="center")

    # Backward lean angle arc (in Y direction)
    if pose.lean_deg > 2:
        arc = Arc((0.0, pose.ankle[1]),
                  width=0.30, height=0.30,
                  angle=90, theta1=0, theta2=pose.lean_deg,
                  color=C_ANGLE, linewidth=2)
        ax.add_patch(arc)
        ax.text(-0.22, pose.ankle[1] + 0.20,
                f"{pose.lean_deg:.0f}° lean\n(heelside)",
                color=C_ANGLE, fontsize=8.5)

    # Pelvis height annotation
    ax.annotate("", xy=(-0.75, pose.pelvis_z),
                xytext=(-0.75, 0),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax.text(-0.82, pose.pelvis_z / 2,
            f"pelvis\n{pose.pelvis_z:.2f} m",
            color="#555", fontsize=8, ha="center")

    # Ankle height annotation
    az = pose.ankle[1]
    ay = pose.ankle[0]
    ax.annotate("", xy=(ay - 0.10, az),
                xytext=(ay - 0.10, 0),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1.0))
    ax.text(ay - 0.18, az / 2 if az > 0.08 else 0.12,
            f"ankle\n{az:.2f} m",
            color="#888", fontsize=7.5, ha="center")

    # Arm straight note
    if pose.lean_deg > 10:
        ax.text(hy - 0.05, hz - 0.14,
                "Arms STRAIGHT\n(never pull!)",
                color="#C0392B", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.8))
    else:
        ax.text(hy - 0.05, hz - 0.14,
                "Arms bent\n(grip handle)",
                color="#7F8C8D", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.8))

    # Board label
    ax.text(0.0, -0.06, "Board: 1.4 m × 0.4 m\n(length along Y visible here)",
            fontsize=7.5, ha="center", color="#8B6914",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

    ax.set_xlim(-0.90, 0.90)
    ax.set_ylim(-0.18, 1.50)
    ax.set_aspect("equal")
    ax.set_xlabel("Y  (← heelside  |  rider faces →  toeside)", fontsize=10)
    ax.set_ylabel("Z  (↑ = up)", fontsize=10)
    ax.set_title(pose.name, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Legend
    ax.scatter([], [], color=C_JOINT, s=60, label="Joint")
    ax.scatter([], [], color=C_COM, s=120, marker="*", label="Centre of Mass")
    ax.fill([], [], color=C_BOARD, alpha=0.85, label="Wakeboard (1.4 m × 0.4 m)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

# ── Shared notes box ────────────────────────────────────────────────────────
notes = (
    "WHAT THE POLICY MUST LEARN  (gap = initial → target)\n"
    "  • Hip pitch:     -0.80 → -0.15 rad  (unfold from cannonball)\n"
    "  • Knee:           1.40 →  0.25 rad  (extend legs, stay softly bent)\n"
    "  • Elbow:          1.00 →  0.10 rad  (straighten arms — CRITICAL: never pull rope)\n"
    "  • Torso lean:      5°  →   22°      (lean heelside to counter 600 N rope)\n"
    "  • Pelvis Z:       0.50 →  0.85 m   (rise from water level to riding height)\n\n"
    "VIEW NOTE: Looking from +X (boat direction). Board full 1.4m length visible here.\n"
    "  Rope pulls INTO PAGE (+X). Rider faces RIGHT (+Y). Backward lean = toward LEFT (-Y = heelside).\n"
    "HEAD NOTE: After 90° yaw, G1 head faces +Y. Should face +X toward boat — deferred until vision phase."
)
fig.text(0.5, 0.01, notes, ha="center", va="bottom", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.6",
                   facecolor="white", edgecolor="#BDC3C7", alpha=0.95),
         fontfamily="monospace")

plt.tight_layout(rect=[0, 0.16, 1, 0.96])
plt.savefig("pose_diagram.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
print("Saved: pose_diagram.png")
plt.close()


# ── PHASE 2 diagram ─────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8), facecolor=C_BG)
ax2.set_facecolor(C_BG)
fig2.suptitle("Phase 2 Goal — Natural Wakeboarding Stance (after successful start)",
              fontsize=13, fontweight="bold")

ax2.axhline(0, color="#7F8C8D", linewidth=1.5, linestyle="--", alpha=0.6)
ax2.fill_between([-0.85, 0.85], -0.12, 0, color="#AED6F1", alpha=0.4)

PHASE2 = Pose2D(
    name="PHASE 2\n(Natural Riding)",
    pelvis_z=0.88,
    hip_pitch=-0.10,
    knee_flex=0.18,
    ankle_pitch=0.05,
    shoulder_pitch=0.30,
    elbow_flex=0.05,
    torso_lean_y=0.24,   # 14° — less lean at stable cruise
    board_front=+0.70,
    board_back=-0.70,
)
PHASE2.draw(ax2)

# Rope symbol
hy, hz = PHASE2.hand[0], PHASE2.hand[1]
ax2.scatter([hy], [hz], s=400, facecolors="white",
            edgecolors=C_ROPE, linewidths=2.5, zorder=10)
ax2.scatter([hy], [hz], s=80, color=C_ROPE, zorder=11)
ax2.text(hy + 0.08, hz + 0.04,
         "⊙ ROPE\n(+X, continuous\ncruise pull)",
         color=C_ROPE, fontsize=9)

ax2.annotate("transition from\nTarget pose here",
             xy=(-0.10, 1.10), xytext=(-0.55, 1.25),
             arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.5),
             fontsize=9, color="#8E44AD")

ax2.set_xlim(-0.90, 0.90)
ax2.set_ylim(-0.18, 1.50)
ax2.set_aspect("equal")
ax2.set_xlabel("Y  (← heelside  |  rider faces →  toeside)", fontsize=10)
ax2.set_ylabel("Z  (↑ = up)", fontsize=10)
ax2.grid(True, alpha=0.3, linewidth=0.5)

phase2_notes = (
    "Phase 2 starts AFTER Phase 1 success (stable ride ≥1.5 s).\n"
    "Goal: maintain natural riding stance at cruise speed.\n"
    "  • Slightly more upright (less lean — rope tension drops at speed)\n"
    "  • Softer knee bend (relaxed long-ride stance)\n"
    "  • Arms straight throughout (handle at hips, never pull)\n"
    "  • Board tracking flat (nose-up phase over — fully planing)"
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
print("  pose_diagram.png        — initial vs target (from +X / boat-side view)")
print("  pose_diagram_phase2.png — phase 2 natural riding stance")
