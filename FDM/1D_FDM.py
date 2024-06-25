import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# domain
xMin = 0
xMax = 10
xPoints = 100
xLength = xMax - xMin
x = np.linspace(xMin, xMax, xPoints)
dx = xLength / (xPoints - 1)

# time
time = 10.0
timeSteps = 1000
dt = time / (timeSteps - 1)

# wave speed
c = 5.0

# spatially varying density
rho = np.zeros(xPoints)
for i in range(xPoints):
    rho[i] = 1.0 - 0.05 * abs(x[i] - 0.5 * xLength)

# stability (CFL condition)
if (c * dt) / dx > 1.0:
    raise ValueError("CLF condition c*dt / dx <= 1.0 is not satisfied!")

# initialize arrays
u = np.zeros(xPoints)
u_old = np.zeros(xPoints)
u_new = np.zeros(xPoints)

# initial condition
x0 = xLength / 2
sigma = 0.5
for i in range(xPoints):
    u[i] = np.exp(-((x[i] - x0) ** 2) / (2 * sigma ** 2))

# set previous the same a start time
u_old[:] = u

# set up figure
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_xlim(xMin, xMax)
ax.set_ylim(-0.5, 1.5)
ax.grid(visible=1)

# update function
def animate(frame):
    global u, u_old, u_new
    # central difference
    for i in range(1, xPoints - 1):
        rho_half_plus = (rho[i] + rho[i + 1]) / 2
        rho_half_minus = (rho[i] + rho[i - 1]) / 2
        term1 = rho_half_plus * (u[i + 1] - u[i])
        term2 = rho_half_minus * (u[i] - u[i - 1])
        u_new[i] = 2 * u[i] - u_old[i] + ((dt ** 2 * c**2) / (rho[i] * dx ** 2)) * (term1 - term2)

    '''
    # boundary condition (Dirichlet)
    u_new[0] = 0
    u_new[-1] = 0
    '''

    # boundary condition (Neumann)
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]

    # update time step
    u_old[:] = u
    u[:] = u_new

    line.set_ydata(u)
    return line,

# animation
ani = animation.FuncAnimation(fig, animate, frames=range(1, timeSteps), blit=True, interval=50)
plt.xlabel('x')
plt.show()