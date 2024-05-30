import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx = 200       # number of spatial points
nt = 1000      # number of time steps
Lx = 10.0      # length of the domain
dx = Lx / (nx - 1)  # spatial step size
dt = 0.01      # time step size
c = 1.0        # wave speed

# Stability condition
if c * dt / dx > 1:
    raise ValueError("The Courant condition is not met: c*dt/dx <= 1")

# Initial conditions
u = np.zeros(nx)
u_new = np.zeros(nx)
u_old = np.zeros(nx)

# Source term
def source_term(x, t):
    return 0  # No external source for now

# Initial conditions for the wave function (u) at t=0
x0 = Lx / 2  # Center of the Gaussian pulse
sigma = 0.5  # Width of the Gaussian pulse

for i in range(nx):
    x = i * dx
    u[i] = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))  # Initial Gaussian pulse
    u_old[i] = u[i]  # Assuming initial velocity is zero, u_old = u

# Prepare the plot
fig, ax = plt.subplots()
x = np.linspace(0, Lx, nx)
line, = ax.plot(x, u)
ax.set_xlim(0, Lx)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Wave propagation')

# Update function for animation
def update(frame):
    global u, u_old, u_new
    t = frame * dt
    for i in range(1, nx-1):
        u_new[i] = (2 * u[i] - u_old[i] +
                    (c * dt / dx)**2 * (u[i+1] - 2 * u[i] + u[i-1]) +
                    dt**2 * source_term(i * dx, t))
    
    # Update u_old and u
    u_old[:] = u[:]
    u[:] = u_new[:]
    
    line.set_ydata(u)
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=20)

# Save the animation as a video file
#ani.save("wave_propagation.mp4", writer='ffmpeg')

plt.show()
