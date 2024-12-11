import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# domain
xMin = -2
xMax = 2
xPoints = 161
xLength = xMax - xMin
x = np.linspace(xMin, xMax, xPoints)
dx = xLength / (xPoints - 1)

# time
time = 8
dt = 5e-3
timeSteps = int(time / dt) + 1

# material parameters
density = 1.0
wavespeed = 1.0

# initial displacement
frequency = 1.0
lamda = wavespeed / frequency
sigma = lamda / 2 / np.pi

# density
rho = np.zeros(xPoints)
for i in range(xPoints):
    rho[i] = density

# create damaged area
#for i in range(10,20):
#    rho[i] = 1e-6


# stability - CFL condition
if (wavespeed * dt) / dx > 1.0:
    raise ValueError("CLF condition c*dt / dx <= 1.0 is not satisfied!")

# initialize arrays
u = np.zeros(xPoints)
u_old = np.zeros(xPoints)
u_new = np.zeros(xPoints)

# initial condition
x0 = 0
for i in range(xPoints):
    u[i] = 2 * np.exp(-((x[i] - x0) ** 2) / 2 / sigma ** 2)

# set previous the same a start time
u_old[:] = u

# set up figure
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.grid(visible=0.5, which="major", linestyle="--")

# update function
def animate(frame):
    global u, u_old, u_new
    # central difference - looping over all points
    for i in range(1, xPoints - 1):
        rho_half_plus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i + 1])))
        rho_half_minus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i - 1])))

        term1 = rho_half_plus * (u[i + 1] - u[i])
        term2 = rho_half_minus * (u[i] - u[i - 1])

        u_new[i] = 2 * u[i] - u_old[i] + ((dt ** 2 * wavespeed**2) / (rho[i] * dx ** 2)) * (term1 - term2)


    # Dirichlet boundary condition
    #u_new[0] = 0
    #u_new[-1] = 0

    # Neumann boundary condition
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]

    # Absorbing boundary condition
    #u_new[0] = u[1] + ((c * dt - dx) / (c * dt + dx)) * (u_new[1] - u[0])
    #u_new[-1] = u[-2] + ((c * dt - dx) / (c * dt + dx)) * (u_new[-2] - u[-1])

    # update time step
    u_old[:] = u
    u[:] = u_new

    line.set_ydata(u)
    return line,

# animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, timeSteps, 100), interval=100)
plt.xlabel('x', fontsize=14)
plt.ylabel('Displacement $u$', fontsize=14)
plt.xlim([xMin, xMax])
plt.ylim([-0.2, 2.2])
plt.show()