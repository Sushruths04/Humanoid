"""Generate a visualization video from rollout_trace.json."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow, Rectangle
import sys

trace_path = sys.argv[1] if len(sys.argv) > 1 else "rollout_trace.json"
out_path   = sys.argv[2] if len(sys.argv) > 2 else "rollout_vis.mp4"

with open(trace_path) as f:
    d = json.load(f)

N = len(d["pelvis_z"])
steps       = d["step"]
pelvis_z    = d["pelvis_z"]
pelvis_x    = d["pelvis_x"]
uprightness = d["uprightness"]
rope_force  = d["rope_force"]
board_pitch = d["board_pitch"]
fell        = d["fell"]
reward      = d["reward"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')

def style_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.set_xlabel('Step', color='#8b949e', fontsize=9)
    if ylim: ax.set_ylim(ylim)
    ax.set_xlim(0, max(steps) + 1)

# --- SIDE VIEW: G1 stick figure on board ---
ax_main = axes[0, 0]
ax_main.set_xlim(-2, 5)
ax_main.set_ylim(-0.3, 2.0)
ax_main.set_aspect('equal')
style_ax(ax_main, 'G1 Side View (Simulated)', 'Height (m)')
ax_main.axhline(0, color='#8b4513', linewidth=2, alpha=0.5)  # ground

board_patch = Rectangle((-0.7, -0.05), 1.4, 0.05, color='#f0c040', alpha=0.9)
ax_main.add_patch(board_patch)

# stick figure lines
torso_line,  = ax_main.plot([], [], 'o-', color='#58a6ff', linewidth=3, markersize=6)
head_dot,    = ax_main.plot([], [], 'o',  color='#f78166', markersize=10)
left_leg,    = ax_main.plot([], [], 'o-', color='#3fb950', linewidth=2, markersize=4)
right_leg,   = ax_main.plot([], [], 'o-', color='#3fb950', linewidth=2, markersize=4)
arm_line,    = ax_main.plot([], [], 'o-', color='#d2a8ff', linewidth=2, markersize=4)
rope_line,   = ax_main.plot([], [], '--', color='#f0c040', linewidth=1.5, alpha=0.7)
force_arrow  = ax_main.annotate('', xy=(0,0), xytext=(0,0),
                                 arrowprops=dict(arrowstyle='->', color='#ff7b72', lw=2))
step_text    = ax_main.text(0.02, 0.97, '', transform=ax_main.transAxes,
                             color='#e6edf3', fontsize=10, va='top')
fell_text    = ax_main.text(0.5, 0.5, '', transform=ax_main.transAxes,
                             color='#ff7b72', fontsize=16, ha='center', va='center', fontweight='bold')

# --- Metric plots ---
ax_pz  = axes[0, 1];  style_ax(ax_pz,  'Pelvis Height', 'm', [0, 1.5])
ax_up  = axes[0, 2];  style_ax(ax_up,  'Uprightness',   '', [0, 1.1])
ax_bp  = axes[1, 0];  style_ax(ax_bp,  'Board Pitch',   'deg', [-50, 70])
ax_rf  = axes[1, 1];  style_ax(ax_rf,  'Rope Force',    'N', [0, 650])
ax_rew = axes[1, 2];  style_ax(ax_rew, 'Step Reward',   '')

line_pz,  = ax_pz.plot([], [], color='#58a6ff', linewidth=2)
line_up,  = ax_up.plot([], [], color='#3fb950', linewidth=2)
line_bp,  = ax_bp.plot([], [], color='#f0c040', linewidth=2)
line_rf,  = ax_rf.plot([], [], color='#ff7b72', linewidth=2)
line_rew, = ax_rew.plot([], [], color='#d2a8ff', linewidth=2)

# reference lines
ax_pz.axhline(0.55, color='#8b949e', linestyle=':', linewidth=1, alpha=0.5, label='crouch')
ax_up.axhline(0.85, color='#8b949e', linestyle=':', linewidth=1, alpha=0.5, label='target')
ax_bp.axhline(0,    color='#8b949e', linestyle=':', linewidth=1, alpha=0.5)

fig.suptitle('G1 Wakeboard Rollout — Stage I (10 km/h)', color='#e6edf3',
             fontsize=14, fontweight='bold', y=1.01)

def update(i):
    pz  = pelvis_z[i]
    px  = pelvis_x[i] - pelvis_x[0]
    up  = uprightness[i]
    bp  = board_pitch[i]
    rf  = rope_force[i]

    # Stick figure from pelvis_z + uprightness
    hip_y   = pz
    hip_x   = px
    knee_y  = hip_y - 0.35 * (1 - 0.3 * up)
    knee_x  = hip_x - 0.1
    foot_y  = 0.0
    foot_x  = hip_x - 0.15
    torso_top_y = hip_y + 0.35 * up
    torso_top_x = hip_x + 0.05 * (1 - up)
    head_y  = torso_top_y + 0.15
    shoulder_y  = torso_top_y - 0.05
    hand_y  = shoulder_y - 0.25
    hand_x  = torso_top_x - 0.35

    torso_line.set_data([foot_x, knee_x, hip_x, torso_top_x],
                        [foot_y, knee_y, hip_y, torso_top_y])
    head_dot.set_data([torso_top_x], [head_y])
    arm_line.set_data([torso_top_x, hand_x], [shoulder_y, hand_y])
    rope_line.set_data([hand_x, hand_x + 1.5], [hand_y, hand_y + 0.1])

    # Board
    board_patch.set_x(foot_x - 0.7)

    # Rope force arrow
    force_arrow.remove()
    ax_main.annotate(
        '', xy=(hand_x + min(rf / 300, 1.5), hand_y),
        xytext=(hand_x, hand_y),
        arrowprops=dict(arrowstyle='->', color='#ff7b72', lw=2)
    )

    step_text.set_text(f'Step {i}  |  H={pz:.2f}m  |  Up={up:.2f}  |  F={rf:.0f}N')
    fell_text.set_text('FELL' if fell[i] else '')

    # Metric lines
    xs = steps[:i+1]
    line_pz.set_data(xs,  pelvis_z[:i+1])
    line_up.set_data(xs,  uprightness[:i+1])
    line_bp.set_data(xs,  board_pitch[:i+1])
    line_rf.set_data(xs,  rope_force[:i+1])
    line_rew.set_data(xs, reward[:i+1])

    return torso_line, head_dot, arm_line, rope_line, step_text, fell_text, \
           line_pz, line_up, line_bp, line_rf, line_rew

ani = animation.FuncAnimation(fig, update, frames=N, interval=80, blit=False)
writer = animation.FFMpegWriter(fps=12, bitrate=1800)
ani.save(out_path, writer=writer, dpi=120)
print(f"Saved: {out_path}  ({N} frames)")
