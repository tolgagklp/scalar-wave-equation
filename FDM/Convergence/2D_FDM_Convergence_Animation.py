import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# domain
xMin = 0
xMax = 1
xPoints = 51
yMin = 0
yMax = 1
yPoints = 51


xLength = xMax - xMin
yLength = yMax - yMin
x = np.linspace(xMin, xMax, xPoints)
y = np.linspace(yMin, yMax, yPoints)
dx = xLength / (xPoints - 1)
dy = yLength / (yPoints - 1)

# time
time = 1.0
timeSteps = 140
dt = 0.01

# wave speed
c = 1.0

# spatially varying density
rho = np.zeros((xPoints, yPoints))
for i in range(xPoints):
    for j in range(yPoints):
        rho[i, j] = 1.0


# stability (CFL condition)
print("CFL value: ", (c * dt) / dx)

if (c * dt) / dx > 1.0 or (c * dt) / dy > 1.0:
    raise ValueError("CLF condition is not satisfied!")

# initialize arrays
u = np.zeros((xPoints, yPoints))
u_old = np.zeros((xPoints, yPoints))
u_new = np.zeros((xPoints, yPoints))

# initial condition
for i in range(xPoints):
    for j in range(yPoints):
        u[i, j] = (x[i] * (x[i] - 1)) * (y[j] * (y[j] - 1))

# set previous the same a start time
u_old[:, :] = u[:, :]

# Prepare the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
surface = ax.plot_surface(X, Y, u.T, cmap='plasma', vmin=-0.06, vmax=0.06)

ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
ax.set_zlim(-0.07, 0.07)

# update function
def animate(frame):
    global u, u_old, u_new
    # central difference
    for i in range(1, xPoints - 1):
        for j in range(1, yPoints - 1):
            x_rho_half_plus = 1 / ((1 / (2 * rho[i,j])) + (1 / (2 * rho[i + 1,j])))
            x_rho_half_minus = 1 / ((1 / (2 * rho[i,j])) + (1 / (2 * rho[i - 1,j])))

            y_rho_half_plus = 1 / ((1 / (2 * rho[i, j])) + (1 / (2 * rho[i, j + 1])))
            y_rho_half_minus = 1 / ((1 / (2 * rho[i, j])) + (1 / (2 * rho[i, j - 1])))

            x_term1 = x_rho_half_plus * (u[i + 1, j] - u[i, j])
            x_term2 = x_rho_half_minus * (u[i, j] - u[i - 1, j])

            y_term1 = y_rho_half_plus * (u[i, j + 1] - u[i, j])
            y_term2 = y_rho_half_minus * (u[i, j] - u[i, j - 1])

            u_new[i, j] = 2 * u[i, j] - u_old[i, j] + ((dt ** 2 * c**2) / (rho[i,j] * dx ** 2)) * (x_term1 - x_term2) + ((dt ** 2 * c**2) / (rho[i,j] * dy ** 2)) * (y_term1 - y_term2)


    # boundary condition (Dirichlet)
    u_new[0, :] = 0
    u_new[:, 0] = 0
    u_new[-1, :] = 0
    u_new[:, -1] = 0

    # update time step
    u_old[:, :] = u[:, :]
    u[:, :] = u_new[:, :]

    # Update surface data
    ax.clear()
    surface = ax.plot_surface(X, Y, u.T, cmap='plasma', vmin=-0.06, vmax=0.06)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_zlim(-0.07, 0.07)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return surface,



# animation
ani = animation.FuncAnimation(fig, animate, frames=timeSteps, blit=False, interval=50, repeat=False)
plt.show()

# safe animation

writer = PillowWriter(fps=10)
ani.save("2D_popagation.gif", writer=writer)