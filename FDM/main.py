import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
domain = [0, 10]
nx = 200        # number of spatial points 200
nt = 500       # number of time steps 500


Lx = domain[1] - domain[0]      # length of the domain
dx = Lx / (nx - 1)  # spatial step size
dt = 0.01       # time step size

# Define wave speed and density as functions of position
def wave_speed(x):
    return 1.0  # Constant wave speed through the domain

def density(x):
    return 1.0  # Constan density through the domain

# Source term function
def source_term(x, t):
    return 0                                           # Homogeneous
    #return np.sin(np.pi * x) * np.cos(2 * np.pi * t)  # Example source term

# Stability condition
if np.max([wave_speed(i*dx) for i in range(nx)]) * dt / dx > 1:
    raise ValueError("The Courant condition is not met: c*dt/dx <= 1")

# Initial conditions
u = np.zeros(nx)
u_new = np.zeros(nx)
u_old = np.zeros(nx)

# Initial condition function
def initial_condition(x):
    x0 = Lx / 2  # Center of the Gaussian pulse
    sigma = 0.5  # Width of the Gaussian pulse
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))  # Modify this for other initial conditions

# def initial_condition(x):
#     x0 = Lx / 2  # Üçgen dalgasının merkezi
#     width = 1.0  # Üçgen dalgasının genişliği
#     return max(0, 1 - abs(x - x0) / (width / 2))



# Apply initial condition
for i in range(nx):
    x = i * dx
    u[i] = initial_condition(x)
    u_old[i] = u[i]  # Assuming initial velocity is zero, u_old = u

# Boundary conditions (Dirichlet)
def apply_boundary_conditions(u):
    u[0] = 0  # Left boundary
    u[-1] = 0  # Right boundary
    return u

# Prepare the plot
fig, ax = plt.subplots()
x = np.linspace(domain[0], domain[1], nx)   
line, = ax.plot(x, u)
ax.set_xlim(domain[0], domain[1])           
ax.set_ylim(-1, 1)                          # Just for the initiation
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Wave propagation')

# Autoscale for y axis
#ax.autoscale(enable=True, axis='y', tight=False)

# Update function for animation
def update(frame):
    global u, u_old, u_new
    t = frame * dt
    for i in range(1, nx-1):
        c = wave_speed(i * dx)
        p = density(i * dx)
        u_new[i] = (2 * u[i] - u_old[i] +
                    (c * dt / dx)**2 * ((u[i+1] - u[i]) / p - (u[i] - u[i-1]) / p) +
                    dt**2 * source_term(i * dx, t))  # Include source term
    
    # Apply boundary conditions
    u_new = apply_boundary_conditions(u_new)

    # Update u_old and u
    u_old[:] = u[:]
    u[:] = u_new[:]
    
    line.set_ydata(u)
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=20)
FuncAnimation()
plt.show()
