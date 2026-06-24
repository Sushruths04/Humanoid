"""Generate a visualization video/GIF from rollout_trace.json."""
import json, numpy as np, sys, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

trace_path = sys.argv[1] if len(sys.argv) > 1 else "rollout_trace.json"
out_path   = sys.argv[2] if len(sys.argv) > 2 else "rollout_vis.gif"

with open(trace_path) as f:
    d = json.load(f)

N           = len(d["pelvis_z"])
steps       = d["step"]
pelvis_z    = d["pelvis_z"]
pelvis_x    = d["pelvis_x"]
uprightness = d["uprightness"]
rope_force  = d["rope_force"]
board_pitch = d["board_pitch"]
fell        = d["fell"]
reward      = d["reward"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)

def style(ax, title, ylabel, ylim=None):
    ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='bold', pad=4)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)
    ax.set_xlabel('Step', color='#8b949e', fontsize=8)
    if ylim: ax.set_ylim(ylim)
    ax.set_xlim(0, max(steps) + 1)

# --- Side view ---
ax_main = axes[0, 0]
ax_main.set_xlim(-1.5, 4.0)
ax_main.set_ylim(-0.2, 2.0)
ax_main.set_aspect('equal')
style(ax_main, 'G1 Side View', 'Height (m)')
ax_main.axhline(0, color='#8b4513', linewidth=3, alpha=0.6)

board_rect = Rectangle((-0.7, -0.06), 1.4, 0.06, color='#f0c040', alpha=0.9, zorder=3)
ax_main.add_patch(board_rect)

body_line,   = ax_main.plot([], [], 'o-', color='#58a6ff', lw=3, ms=5, zorder=4)
head_pt,     = ax_main.plot([], [], 'o',  color='#f78166', ms=12, zorder=5)
arm_line,    = ax_main.plot([], [], 'o-', color='#d2a8ff', lw=2, ms=4, zorder=4)
rope_line,   = ax_main.plot([], [], '--', color='#f0c040', lw=1.5, alpha=0.8, zorder=3)
force_line,  = ax_main.plot([], [], '-',  color='#ff7b72', lw=3, zorder=4)
info_txt     = ax_main.text(0.02, 0.97, '', transform=ax_main.transAxes,
                             color='#e6edf3', fontsize=9, va='top', family='monospace')

# --- Metric plots ---
ax_pz  = axes[0, 1]; style(ax_pz,  'Pelvis Height', 'm',   [0.3, 2.0])
ax_up  = axes[0, 2]; style(ax_up,  'Uprightness',   '',    [0, 1.1])
ax_bp  = axes[1, 0]; style(ax_bp,  'Board Pitch',   'deg', [-60, 75])
ax_rf  = axes[1, 1]; style(ax_rf,  'Rope Force',    'N',   [0, 660])
ax_rew = axes[1, 2]; style(ax_rew, 'Step Reward',   '')

ax_pz.axhline(0.55, color='#30363d', ls=':', lw=1)
ax_up.axhline(0.85, color='#30363d', ls=':', lw=1)
ax_bp.axhline(0,    color='#30363d', ls=':', lw=1)

ln_pz,  = ax_pz.plot([], [], color='#58a6ff', lw=2)
ln_up,  = ax_up.plot([], [], color='#3fb950', lw=2)
ln_bp,  = ax_bp.plot([], [], color='#f0c040', lw=2)
ln_rf,  = ax_rf.plot([], [], color='#ff7b72', lw=2)
ln_rew, = ax_rew.plot([], [], color='#d2a8ff', lw=2)

cur_pz,  = ax_pz.plot([], [], 'o', color='#58a6ff', ms=5)
cur_up,  = ax_up.plot([], [], 'o', color='#3fb950', ms=5)
cur_bp,  = ax_bp.plot([], [], 'o', color='#f0c040', ms=5)
cur_rf,  = ax_rf.plot([], [], 'o', color='#ff7b72', ms=5)
cur_rew, = ax_rew.plot([], [], 'o', color='#d2a8ff', ms=5)

fig.suptitle('G1 Wakeboard — Stage I Policy Rollout (model_980, 10 km/h)',
             color='#e6edf3', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

def update(i):
    pz  = pelvis_z[i]
    px  = pelvis_x[i] - pelvis_x[0]
    up  = max(0.1, uprightness[i])
    rf  = rope_force[i]

    # Stick figure
    foot_x, foot_y     = px, 0.0
    knee_x = px - 0.12 * (1.5 - up)
    knee_y = pz - 0.38
    hip_x,  hip_y      = px, pz
    torso_x = px + 0.06 * (1 - up)
    torso_y = pz + 0.38 * up
    head_x  = torso_x
    head_y  = torso_y + 0.16
    hand_x  = torso_x - 0.35
    hand_y  = torso_y - 0.12

    body_line.set_data([foot_x, knee_x, hip_x, torso_x], [foot_y, knee_y, hip_y, torso_y])
    head_pt.set_data([head_x], [head_y])
    arm_line.set_data([torso_x, hand_x], [torso_y - 0.05, hand_y])
    rope_line.set_data([hand_x, hand_x + 1.8], [hand_y, hand_y + 0.05])

    # Force arrow (line with arrowhead via scatter)
    flen = min(rf / 350, 1.2)
    force_line.set_data([hand_x, hand_x + flen], [hand_y, hand_y])

    # Board
    board_rect.set_x(foot_x - 0.7)

    info_txt.set_text(
        f'Step {i:>3d} | H={pz:.2f}m | Up={up:.2f} | F={rf:.0f}N'
        + (' | FELL' if fell[i] else '')
    )

    xs = steps[:i+1]
    ln_pz.set_data(xs, pelvis_z[:i+1]);     cur_pz.set_data([steps[i]], [pz])
    ln_up.set_data(xs, uprightness[:i+1]);  cur_up.set_data([steps[i]], [up])
    ln_bp.set_data(xs, board_pitch[:i+1]);  cur_bp.set_data([steps[i]], [board_pitch[i]])
    ln_rf.set_data(xs, rope_force[:i+1]);   cur_rf.set_data([steps[i]], [rf])
    ln_rew.set_data(xs, reward[:i+1]);      cur_rew.set_data([steps[i]], [reward[i]])

    return (body_line, head_pt, arm_line, rope_line, force_line, board_rect, info_txt,
            ln_pz, ln_up, ln_bp, ln_rf, ln_rew,
            cur_pz, cur_up, cur_bp, cur_rf, cur_rew)

# Subsample to 150 frames max for GIF size
frame_idx = list(range(0, N, max(1, N // 150)))
ani = animation.FuncAnimation(fig, update, frames=frame_idx, interval=80, blit=False)

if out_path.endswith('.mp4') and shutil.which('ffmpeg'):
    writer = animation.FFMpegWriter(fps=15, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=110)
else:
    out_path = out_path.replace('.mp4', '.gif')
    ani.save(out_path, writer='pillow', fps=15, dpi=90)

print(f"Saved: {out_path}  ({len(frame_idx)} frames, {N} total steps)")
print(f"Stats: pelvis_max={max(pelvis_z):.2f}m  fell={sum(fell):.0f}/{N}  force_avg={sum(rope_force)/N:.0f}N")
