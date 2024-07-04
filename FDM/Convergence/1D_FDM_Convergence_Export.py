import numpy as np
import matplotlib.pyplot as plt
import functionsConvergence as fC

# domain
xMin = 0
xMax = 10
#xPoints = [101, 251, 501, 1001, 2501, 5001]
xPoints = [51, 101, 201, 401, 601, 1201]

# time
dt = 0.001
timeSteps = 2500

# wave speed
c = 1.0

# density
rho_0 = 1.0


for i_xPoints in xPoints:
    xLength = xMax - xMin
    x = np.linspace(xMin, xMax, i_xPoints)
    dx = xLength / (i_xPoints - 1)


    # spatially varying density
    rho = np.zeros(i_xPoints)
    rho[:] = rho_0

    # stability (CFL condition)
    if (c * dt) / dx > 1.0:
        raise ValueError("CLF condition c*dt / dx <= 1.0 is not satisfied!")

    # initialize arrays
    u = np.zeros(i_xPoints)
    u_old = np.zeros(i_xPoints)
    u_new = np.zeros(i_xPoints)
    u_ref = np.zeros(i_xPoints)

    # initial condition
    x0 = xLength / 2
    sigma = 0.5
    for i in range(0, i_xPoints):
        u[i] = fC.analyticGaussianSolution(x[i], x0, 0, c, sigma)
    # set previous the same a start time
    u_old[:] = u

    # time stepping
    for j in range(1, timeSteps+1):

        # central difference
        for i in range(1, i_xPoints - 1):
            rho_half_plus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i + 1])))
            rho_half_minus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i - 1])))
            term1 = rho_half_plus * (u[i + 1] - u[i])
            term2 = rho_half_minus * (u[i] - u[i - 1])

            u_new[i] = 2 * u[i] - u_old[i] + ((dt ** 2 * c ** 2) / (rho[i] * dx ** 2)) * (term1 - term2)

            u_ref[i] = fC.analyticGaussianSolution(x[i], x0, j*dt, c, sigma)

        # boundary condition (Neumann)
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        # update time step
        u_old[:] = u
        u[:] = u_new

    np.save(f"x_xPoints_{i_xPoints}", x)
    np.save(f"u_FDM_xPoints_{i_xPoints}_dt_{dt}_timeSteps_{timeSteps}", u)
    np.save(f"u_ref_xPoints_{i_xPoints}_dt_{dt}_timeSteps_{timeSteps}", u_ref)

#plt.plot(x, u, 'r--')
#plt.plot(x, u_ref, 'b-')
#plt.show()