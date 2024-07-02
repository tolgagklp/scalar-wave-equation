import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Parameters
domain = [0, 10]
nx = 100  # number of spatial points in x direction
ny = 100  # number of spatial points in y direction
nt = 1000  # number of time steps (reduced for faster animation)

Lx = domain[1] - domain[0]  # length of the domain in x direction
Ly = domain[1] - domain[0]  # length of the domain in y direction
dx = Lx / (nx - 1)  # spatial step size in x direction
dy = Ly / (ny - 1)  # spatial step size in y direction
dt = 0.01  # time step size


# Define wave speed and density as functions of position
def wave_speed(x, y):
    return 4.5  # Constant wave speed through the domain


def density(x, y):
    return 1.0  # Constant density through the domain


# Source term function
def source_term(x, y, t):
    return 0  # Homogeneous


# Stability condition
if np.max([wave_speed(i * dx, j * dy) for i in range(nx) for j in range(ny)]) * dt / min(dx, dy) > 1:
    raise ValueError("The Courant condition is not met: c*dt/min(dx,dy) <= 1")

# Initial conditions
u = np.zeros((nx, ny))
u_new = np.zeros((nx, ny))
u_old = np.zeros((nx, ny))


# Initial condition function
def initial_condition(x, y):
    x0, y0 = Lx / 2, Ly / 2  # Center of the Gaussian pulse
    sigma = 0.5  # Width of the Gaussian pulse
    return 2 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# Apply initial condition
for i in range(nx):
    for j in range(ny):
        x = i * dx
        y = j * dy
        u[i, j] = initial_condition(x, y)
        u_old[i, j] = u[i, j]  # Assuming initial velocity is zero, u_old = u

'''
# Boundary conditions (Dirichlet)
def apply_boundary_conditions(u):
    u[0, :] = 0  # Left boundary
    u[-1, :] = 0  # Right boundary
    u[:, 0] = 0 # Bottom boundary
    u[:, -1] = 0  # Top boundary
    return u
'''
# Boundary conditions (Neumann)
def apply_boundary_conditions(u):
    u[0,0] = u[1,1]
    u[0,-1] = u[1,-2]
    u[-1,0] = u[-2,1]
    u[-1,-1] = u[-2,-2]
    u[0, 1:-1] = u[1, 1:-1]       # Left boundary
    u[-1, 1:-1] = u[-2, 1:-1]     # Right boundary
    u[1:-1, 0] = u[1:-1, 1]       # Bottom boundary
    u[1:-1, -1] = u[1:-1, -2]     # Top boundary
    return u


# Prepare the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(domain[0], domain[1], nx)
y = np.linspace(domain[0], domain[1], ny)
X, Y = np.meshgrid(x, y)
surface = ax.plot_surface(X, Y, u.T, cmap='seismic', vmin=-1, vmax=1)

ax.set_xlim(domain[0], domain[1])
ax.set_ylim(domain[0], domain[1])
ax.set_zlim(-2, 2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Amplitude')
ax.set_title('Wave propagation')


# Update function for animation
def update(frame):
    global u, u_old, u_new
    t = frame * dt
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            c = wave_speed(i * dx, j * dy)
            p = density(i * dx, j * dy)
            p_plus_x = density((i + 1) * dx, j * dy)
            p_minus_x = density((i - 1) * dx, j * dy)
            p_plus_y = density(i * dx, (j + 1) * dy)
            p_minus_y = density(i * dx, (j - 1) * dy)
            u_new[i, j] = (2 * u[i, j] - u_old[i, j] +
                           (c * dt / dx) ** 2 * (
                                       (u[i + 1, j] - u[i, j]) / p_plus_x - (u[i, j] - u[i - 1, j]) / p_minus_x) +
                           (c * dt / dy) ** 2 * (
                                       (u[i, j + 1] - u[i, j]) / p_plus_y - (u[i, j] - u[i, j - 1]) / p_minus_y) +
                           dt ** 2 * source_term(i * dx, j * dy, t))  # Include source term

    # Apply boundary conditions
    u_new = apply_boundary_conditions(u_new)

    # Update u_old and u
    u_old[:, :] = u[:, :]
    u[:, :] = u_new[:, :]

    # Update surface data
    ax.clear()
    ax.plot_surface(X, Y, u.T, cmap='seismic', vmin=-1, vmax=1)
    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[0], domain[1])
    ax.set_zlim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Amplitude')
    ax.set_title('Wave propagation')

    return surface


# Create animation
ani = FuncAnimation(fig, update, frames=nt, blit=False, interval=20)
plt.show()