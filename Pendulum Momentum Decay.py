import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters (tweak these)
# -----------------------------
g = 9.81
L = 2.0
mu = 0.10            # damping strength

theta0 = np.pi / 3   # initial angle
omega0 = 0.0         # initial angular velocity

dt = 0.01
t_end = 20.0
skip = 2             # animation frame skip (higher = faster animation)

trail_seconds = 6.0  # how much trail to show behind the bob
# -----------------------------

def theta_dd(theta, omega):
    return -mu * omega - (g / L) * np.sin(theta)

# Time grid
t = np.arange(0, t_end + dt, dt)

# Integrate (semi-implicit Euler for better stability than plain Euler)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
theta[0] = theta0
omega[0] = omega0

for i in range(1, len(t)):
    a = theta_dd(theta[i-1], omega[i-1])
    omega[i] = omega[i-1] + a * dt
    theta[i] = theta[i-1] + omega[i] * dt

# Convert to bob position
x = L * np.sin(theta)
y = -L * np.cos(theta)

# -----------------------------
# Figure layout: main + inset
# -----------------------------
fig = plt.figure(figsize=(9, 6))
ax_xy = fig.add_subplot(1, 1, 1)
ax_xy.set_aspect("equal", "box")
ax_xy.set_xlim(-1.2 * L, 1.2 * L)
ax_xy.set_ylim(-1.2 * L, 0.25 * L)
ax_xy.set_title("Damped Pendulum: bob/rod + trail, with phase-flow inset")
ax_xy.grid(True)

# Rod + bob
rod, = ax_xy.plot([], [], lw=2)
bob, = ax_xy.plot([], [], "o", ms=10)

# Bob trail (path in x/y)
trail_line, = ax_xy.plot([], [], lw=1, alpha=0.8)

# Draw pivot
ax_xy.plot([0], [0], "o", ms=4)

# Inset axes for phase portrait (theta vs omega), centered on origin
ax_phase = ax_xy.inset_axes([0.62, 0.55, 0.35, 0.35])  # [x0, y0, w, h] in axes fraction
ax_phase.set_title("Phase flow: (θ, ω)", fontsize=9)
ax_phase.set_xlabel("θ", fontsize=8)
ax_phase.set_ylabel("ω", fontsize=8)
ax_phase.tick_params(labelsize=8)
ax_phase.grid(True)

# Phase portrait bounds (centered on origin)
th_max = np.pi
om_max = max(2.0, float(np.max(np.abs(omega))) * 1.05)
ax_phase.set_xlim(-th_max, th_max)
ax_phase.set_ylim(-om_max, om_max)

# Vector field (flow)
TH, OM = np.meshgrid(
    np.linspace(-th_max, th_max, 23),
    np.linspace(-om_max, om_max, 23),
)
DTH = OM
DOM = -mu * OM - (g / L) * np.sin(TH)
# Vector field (flow) on phase plane
TH, OM = np.meshgrid(
    np.linspace(-th_max, th_max, 60),
    np.linspace(-om_max, om_max, 60),
)

DTH = OM
DOM = -mu * OM - (g / L) * np.sin(TH)

speed = np.hypot(DTH, DOM)  # magnitude of the vector field

# 1) Streamlines = "vector pathways"
strm = ax_phase.streamplot(
    TH, OM, DTH, DOM,
    color=speed,          # color by magnitude
    linewidth=1.0,
    density=1.2,          # increase for more lines
    arrowsize=1.2,
)

# Colorbar for magnitude (speed)
cbar = fig.colorbar(strm.lines, ax=ax_phase, fraction=0.046, pad=0.04)
cbar.set_label("Field magnitude |(θ̇, ω̇)|", fontsize=8)
cbar.ax.tick_params(labelsize=8)

# 2) OPTIONAL: normalized quiver on top (uniform arrow length)
# Normalize vectors for direction-only arrows
eps = 1e-12
U = DTH / (speed + eps)
V = DOM / (speed + eps)

ax_phase.quiver(
    TH[::4, ::4], OM[::4, ::4],   # thin the grid so it’s not a porcupine
    U[::4, ::4], V[::4, ::4],
    speed[::4, ::4],              # color still represents magnitude
    angles="xy",
    scale_units="xy",
    scale=18,                      # controls arrow length
    width=0.004,
    alpha=0.9
)


# Phase trajectory + moving point
phase_line, = ax_phase.plot([], [], lw=1.5)
phase_pt, = ax_phase.plot([], [], "o", ms=5)

# Trail lengths in frames
trail_len = max(1, int(trail_seconds / (dt * skip)))

def init():
    rod.set_data([], [])
    bob.set_data([], [])
    trail_line.set_data([], [])
    phase_line.set_data([], [])
    phase_pt.set_data([], [])
    return rod, bob, trail_line, phase_line, phase_pt

def update(frame):
    i = frame * skip
    if i >= len(t):
        i = len(t) - 1

    # --- x/y pendulum ---
    rod.set_data([0, x[i]], [0, y[i]])
    bob.set_data([x[i]], [y[i]])

    # bob trail (recent history)
    j0 = max(0, i - trail_len)
    trail_line.set_data(x[j0:i+1], y[j0:i+1])

    # --- phase portrait trail + point (θ, ω) ---
    phase_line.set_data(theta[:i+1], omega[:i+1])
    phase_pt.set_data([theta[i]], [omega[i]])

    return rod, bob, trail_line, phase_line, phase_pt

ani = FuncAnimation(
    fig,
    update,
    frames=(len(t) // skip),
    init_func=init,
    interval=dt * 1000 * skip,
    blit=True
)

plt.show()
