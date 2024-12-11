import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# domain
xMin = 0
xMax = 10
xPoints = 200

yMin = 0
yMax = 10
yPoints = 200

xLength = xMax - xMin
yLength = yMax - yMin
x = np.linspace(xMin, xMax, xPoints)
y = np.linspace(yMin, yMax, yPoints)
dx = xLength / (xPoints - 1)
dy = yLength / (yPoints -1)

# time
time = 2.0
dt = 1e-2
timeSteps = int(time / dt) + 1

# material parameters
density = 1.0
wavespeed = 1.0

# initial displacement
frequency = 1.0
lamda = wavespeed / frequency
sigma = lamda / 2 / np.pi

# density
rho = np.zeros((xPoints, yPoints))
for i in range(xPoints):
    for j in range(yPoints):
        rho[i, j] = density

# create damaged area
#for i in range(20,80):
#    for j in range(20,80):
#        rho[i,j] = 1e-5

# stability - CFL condition
if (wavespeed * dt) / dx > 1.0 or (wavespeed * dt) / dy > 1.0:
    raise ValueError("CLF condition is not satisfied!")
print("CFL value: ", (wavespeed * dt) / dx)

# initialize arrays
u = np.zeros((xPoints, yPoints))
u_old = np.zeros((xPoints, yPoints))
u_new = np.zeros((xPoints, yPoints))

# initial condition
x0 = xLength / 2
y0 = yLength / 2
for i in range(xPoints):
    for j in range(yPoints):
        u[i, j] = 2*np.exp(-((x[i] - x0) ** 2 + (y[j] - y0) ** 2) / (2 * sigma ** 2))

# set previous the same a start time
u_old[:, :] = u[:, :]

# set up figure
fig, ax = plt.subplots()
X, Y = np.meshgrid(x, y)
color_mesh = ax.pcolormesh(X, Y, u.T, shading='auto', cmap='seismic', vmin=-1, vmax=1)
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
fig.colorbar(color_mesh, ax=ax)

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

            u_new[i, j] = 2 * u[i, j] - u_old[i, j] + ((dt ** 2 * wavespeed**2) / (rho[i,j] * dx ** 2)) * (x_term1 - x_term2) + ((dt ** 2 * wavespeed**2) / (rho[i,j] * dy ** 2)) * (y_term1 - y_term2)


    # Dirichlet boundary condition
    u_new[0, :] = 0
    u_new[:, 0] = 0
    u_new[-1, :] = 0
    u_new[:, -1] = 0


    # Neumann boundary condition
    # corners
    #u_new[0, 0] = u_new[1, 1]
    #u_new[0, -1] = u_new[1, -2]
    #u_new[-1, 0] = u_new[-2, 1]
    #u_new[-1, -1] = u_new[-2, -2]
    # edges
    #u_new[0, 1:-1] = u_new[1, 1:-1]
    #u_new[-1, 1:-1] = u_new[-2, 1:-1]
    #u_new[1:-1, 0] = u_new[1:-1, 1]
    #u_new[1:-1, -1] = u_new[1:-1, -2]


    # update time step
    u_old[:, :] = u[:, :]
    u[:, :] = u_new[:, :]

    color_mesh.set_array(u.T.ravel())
    return color_mesh,


# animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, timeSteps, 100), blit=True, interval=100.0)
plt.xlabel('x')
plt.ylabel('y')
plt.show()