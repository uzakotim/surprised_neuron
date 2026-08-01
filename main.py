import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------------------------------
# "Surprised Neuron" with PID Homeostasis
# -------------------------------------------------------

dt = 0.02
T = 15
t = np.arange(0, T, dt)

RESTING = -65.0          # mV
TARGET = RESTING

# Neuron state
V = RESTING
velocity = 0.0

# Surprise pulse
stimulus = np.zeros_like(t)
stimulus[(t > 2.0) & (t < 2.6)] = 120

# PID parameters (negative feedback)
Kp = 1.4
Ki = 0.45
Kd = 0.8

integral = 0
prev_error = 0

potential_history = []
pid_history = []

for i in range(len(t)):
    error = TARGET - V

    integral += error * dt
    derivative = (error - prev_error) / dt
    prev_error = error

    # PID output
    pid = Kp * error + Ki * integral + Kd * derivative

    # Simple damped neuron dynamics
    acceleration = (
        stimulus[i]
        + pid
        - 0.7 * velocity
        - 0.12 * (V - RESTING)
    )

    velocity += acceleration * dt
    V += velocity * dt

    potential_history.append(V)
    pid_history.append(pid)

potential_history = np.array(potential_history)
pid_history = np.array(pid_history)

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

fig = plt.figure(figsize=(10, 6))

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# Cartoon neuron
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect("equal")
ax1.axis("off")

body = plt.Circle((0, 0), 0.6, color="gold", ec="black", lw=2)
ax1.add_patch(body)

# Eyes
left_eye = plt.Circle((-0.18, 0.15), 0.08, color="white", ec="black")
right_eye = plt.Circle((0.18, 0.15), 0.08, color="white", ec="black")
left_pupil = plt.Circle((-0.18, 0.15), 0.03, color="black")
right_pupil = plt.Circle((0.18, 0.15), 0.03, color="black")

ax1.add_patch(left_eye)
ax1.add_patch(right_eye)
ax1.add_patch(left_pupil)
ax1.add_patch(right_pupil)

mouth, = ax1.plot([], [], lw=3)

# Dendrites
for ang in np.linspace(0, 2*np.pi, 10, endpoint=False):
    x = [0.6*np.cos(ang), 1.0*np.cos(ang)]
    y = [0.6*np.sin(ang), 1.0*np.sin(ang)]
    ax1.plot(x, y)

ax1.set_title("Surprised Neuron")

# Voltage plot
ax2.set_xlim(0, T)
ax2.set_ylim(RESTING - 5, RESTING + 35)
ax2.set_ylabel("Membrane Potential (mV)")
ax2.grid(True)

line_v, = ax2.plot([], [], lw=2)
point_v, = ax2.plot([], [], "o", ms=8)

# PID plot
ax3.set_xlim(0, T)
ax3.set_ylim(np.min(pid_history)-5, np.max(pid_history)+5)
ax3.grid(True)
ax3.set_ylabel("PID Output")
ax3.set_xlabel("Time (s)")

line_pid, = ax3.plot([], [], lw=2)

status = ax1.text(
    0,
    -1.45,
    "",
    ha="center",
    fontsize=13,
    weight="bold"
)

def mouth_shape(opening):
    theta = np.linspace(-np.pi/2, np.pi/2, 40)
    r = 0.18 + opening
    x = r*np.cos(theta)
    y = -0.25 + 0.7*r*np.sin(theta)
    return x, y

def init():
    line_v.set_data([], [])
    line_pid.set_data([], [])
    return line_v, line_pid

def animate(frame):

    v = potential_history[frame]

    excitement = np.clip((v - RESTING)/25.0, 0, 1)

    # Eyes widen
    left_eye.radius = 0.08 + 0.05*excitement
    right_eye.radius = 0.08 + 0.05*excitement

    # Pupils move upward
    left_pupil.center = (-0.18, 0.15 + 0.04*excitement)
    right_pupil.center = (0.18, 0.15 + 0.04*excitement)

    # Mouth opens
    x, y = mouth_shape(0.14*excitement)
    mouth.set_data(x, y)

    if excitement > 0.7:
        status.set_text("😲 WHOA!!")
    elif excitement > 0.25:
        status.set_text("😮 Surprised...")
    else:
        status.set_text("😌 Calm again")

    line_v.set_data(t[:frame], potential_history[:frame])
    point_v.set_data([t[frame]], [potential_history[frame]])

    line_pid.set_data(t[:frame], pid_history[:frame])

    return (
        line_v,
        point_v,
        line_pid,
        mouth,
        left_eye,
        right_eye,
        left_pupil,
        right_pupil,
        status,
    )

ani = FuncAnimation(
    fig,
    animate,
    frames=len(t),
    init_func=init,
    interval=20,
    blit=False,
)

plt.tight_layout()
plt.show()
