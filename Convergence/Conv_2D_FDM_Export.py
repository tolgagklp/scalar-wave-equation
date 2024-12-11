import numpy as np
import functionsConvergence as fC

# domain
xMin = 0
xMax = 1
xLength = xMax - xMin

yMin = 0
yMax = 1
yLength = yMax - yMin

# material parameters
wavespeed = 1.0
density = 1.0

# time
timeSteps = 1400
dt = 0.0005
time = 0.7

# number of points in each direction for convergence
nPoints = [11, 21, 41, 81, 161]

for iPoints in nPoints:

    x = np.linspace(xMin, xMax, iPoints)
    y = np.linspace(yMin, yMax, iPoints)
    dx = xLength / (iPoints - 1)
    dy = yLength / (iPoints - 1)

    # spatially varying density
    rho = np.zeros((iPoints, iPoints))
    rho[:, :] = density


    # stability (CFL condition)
    if (wavespeed * dt) / dx > 1.0 or (wavespeed * dt) / dy > 1.0:
        raise ValueError("CLF condition is not satisfied!")
    print("CFL value: ", (wavespeed * dt) / dx)

    # initialize arrays
    u = np.zeros((iPoints, iPoints))
    u_old = np.zeros((iPoints, iPoints))
    u_new = np.zeros((iPoints, iPoints))

    # inital condition
    for i in range(1, iPoints-1):
        for j in range(1, iPoints-1):
            u[i, j] = (x[i] * (x[i] - 1)) * (y[j] * (y[j] - 1))

    # set previous the same a start time
    u_old[:, :] = u[:, :]

    # time stepping
    for k in range(1, timeSteps+1):
        # FDM
        for i in range(1, iPoints - 1):
            for j in range(1, iPoints - 1):
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

        # update time step
        u_old[:, :] = u[:, :]
        u[:, :] = u_new[:, :]

    u_ref = np.zeros((iPoints, iPoints))
    X,Y = np.meshgrid(x,y)

    # compute analytical solution
    u_ref = fC.analyticSolution2D(X, Y, time)

    # save results
    np.save(f"xPoints_{iPoints}", x)
    np.save(f"yPoints_{iPoints}", y)
    np.save(f"u_FDM_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}", u)
    np.save(f"u_ref_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}", u_ref)