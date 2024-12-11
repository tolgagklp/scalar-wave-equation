import numpy as np
import functionsConvergence as fC
import time
from scipy.integrate import trapezoid

# discretization for convergence - same to FEM
xPoints = [80, 120, 160, 240, 320, 480, 640, 1280, 2560]
delta_x = [16/80, 16/120, 16/160, 16/240, 16/320, 16/480, 16/640, 16/1280, 16/2560]

for i_xP in range(len(xPoints)):

    # start time
    start = time.time()

    # domain
    xMin = -8.0
    xMax = 8.0
    i_xPoints = xPoints[i_xP] + 1
    xLength = xMax - xMin
    x = np.linspace(xMin, xMax, i_xPoints)
    dx = xLength / (i_xPoints - 1)

    # time
    Tmax = 8
    dt = 5e-4
    timeSteps = int(Tmax / dt) + 1

    # material parameters
    wavespeed = 1.0
    density = 1.0

    # initial displacement
    frequency = 1.0
    lamda = wavespeed / frequency
    sigma = lamda / 2 / np.pi


    # density
    rho = np.zeros(i_xPoints)
    rho[:] = density

    # stability - CFL condition
    if (wavespeed * dt) / dx > 1.0:
        raise ValueError("CLF condition c*dt / dx <= 1.0 is not satisfied!")

    # initialize arrays
    u = np.zeros(i_xPoints)
    u_old = np.zeros(i_xPoints)
    u_new = np.zeros(i_xPoints)
    u_ref = np.zeros(i_xPoints)

    # initial condition
    x0 = 0
    for i in range(0, i_xPoints):
        u[i] = fC.analyticGaussianSolution(x[i], x0, 0, wavespeed, sigma)

    # set previous timestep the same a start time
    u_old[:] = u

    # time stepping - looping over all timesteps
    T = 0
    for j in range(1, timeSteps+1):
        T += dt

        # central difference - looping over all points
        for i in range(1, i_xPoints - 1):
            rho_half_plus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i + 1])))
            rho_half_minus = 1 / ((1 / (2 * rho[i])) + (1 / (2 * rho[i - 1])))
            term1 = rho_half_plus * (u[i + 1] - u[i])
            term2 = rho_half_minus * (u[i] - u[i - 1])

            u_new[i] = 2 * u[i] - u_old[i] + ((dt ** 2 * wavespeed ** 2) / (rho[i] * dx ** 2)) * (term1 - term2)

        # Dirichlet boundary condition
        # u_new[0] = 0
        # u_new[-1] = 0

        # Neumann boundary condition
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        # Absorbing boundary condition
        # u_new[0] = u[1] + ((c * dt - dx) / (c * dt + dx)) * (u_new[1] - u[0])
        # u_new[-1] = u[-2] + ((c * dt - dx) / (c * dt + dx)) * (u_new[-2] - u[-1])

        # update time step
        u_old[:] = u
        u[:] = u_new

    # end time
    end = time.time()

    # run time
    time_s = end - start

    # compute L2 error for time T
    error = np.zeros(i_xPoints)
    ref = np.zeros(i_xPoints)
    for i_X in range(len(x)):

        # analytical solution for time T and same initial displacment 
        ref_uhat = fC.analyticGaussianSolution(x[i_X], 0, T, wavespeed, sigma)

        error[i_X] = (u[i_X] - ref_uhat) ** 2
        ref[i_X] = ref_uhat ** 2

    L2 = np.sqrt(trapezoid(error, x, dx=delta_x[i_xP]) / trapezoid(ref, x, dx=delta_x[i_xP]))

    # print for pre-checking results
    #print(T)
    #print(i_xPoints)
    #print(time_s)
    #print(L2)

    # exporting results
    #np.save(f"FDM_{i_xPoints - 1}_time", time_s)
    #np.save(f"FDM_{i_xPoints - 1}_L2", L2)
    #np.save(f"FDM_{i_xPoints - 1}_X", x)
