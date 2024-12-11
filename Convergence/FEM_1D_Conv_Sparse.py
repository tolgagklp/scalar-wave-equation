import numpy as np
import numpy.polynomial.legendre as geek
import functionsConvergence as fC
import time
from scipy.integrate import trapezoid
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# linear basis functions and their derivatives
def basis_function_left(x, node1, node2):
    return (node2 - x) / (node2 - node1)
def basis_function_right(x, node1, node2):
    return (x - node1) / (node2 - node1)
def basis_function_left_derivative(node1, node2):
    return -1.0 / (node2 - node1)
def basis_function_right_derivative(node1, node2):
    return 1.0 / (node2 - node1)


Elements = [80, 120, 160, 240, 320, 480, 640, 1280, 2560]
delta_x = [16/80, 16/120, 16/160, 16/240, 16/320, 16/480, 16/640, 16/1280, 16/2560]


for i_El in range(len(Elements)):

    start = time.time()

    # material parameters
    density = 1.0
    wavespeed = 1.0

    # initial displacement
    frequency = 1.0
    lambda_dom = wavespeed / frequency
    sigma = lambda_dom / (2 * np.pi)


    # computational domain
    xMin = -8.0
    xMax = 8.0

    # interface to fictitious domain
    interface = 8
    alphaFCM = 1e-6

    # specifying discretization in space
    n = Elements[i_El] + 1
    nodes = np.linspace(xMin, xMax, n)

    # time
    deltaT = 5e-4
    Tmax = 5
    nTimesteps = int(Tmax / deltaT) + 1
    print(nTimesteps)

    # initializing global system matrices
    Mg = lil_matrix((n, n))
    Kg = lil_matrix((n, n))

    # vectors for projecting the initial conditions
    Fint_0 = np.zeros((n, 1))
    Fint_m1 = np.zeros((n, 1))

    # integration rule
    nGP = 2
    GP = geek.leggauss(nGP)[0]
    GP_weights = geek.leggauss(nGP)[1]

    # loop over all elements
    nEl = n - 1
    for iEl in range(0, nEl):
        # Element nodes
        node1 = nodes[iEl]
        node2 = nodes[iEl+1]

        # Element matrices and load vectors
        Me = np.zeros((2, 2))
        Ke = np.zeros((2, 2))
        Fint_0e = np.zeros((2, 1))
        Fint_m1e = np.zeros((2, 1))

        # Gauss points mapped to element
        detJacobian = (node2 - node1) / 2.0
        GP_global = [node1 + 0.5 * (node2 - node1) * (gp + 1) for gp in GP]
        GP_weights_scaled = GP_weights * (node2 - node1) / 2.0

        # loop over Gauss points
        for iGP, xGP in enumerate(GP_global):
            alphaGP = 1 if (xGP < interface) and (xGP > -interface) else alphaFCM

            # Evaluate basis functions and their derivatives at xGP
            BasisFunc_left = basis_function_left(xGP, node1, node2)
            BasisFunc_right = basis_function_right(xGP, node1, node2)
            DerBasisFunc_left = basis_function_left_derivative(node1, node2)
            DerBasisFunc_right = basis_function_right_derivative(node1, node2)

            # projection for initial conditions
            Fint_0e[0] += GP_weights_scaled[iGP] * fC.analyticGaussianSolution(xGP, 0, sigma, wavespeed, 0) * BasisFunc_left
            Fint_0e[1] += GP_weights_scaled[iGP] * fC.analyticGaussianSolution(xGP, 0, sigma, wavespeed, 0) * BasisFunc_right
            Fint_m1e[0] += GP_weights_scaled[iGP] * fC.analyticGaussianSolution(xGP, 0, sigma, wavespeed, -deltaT) * BasisFunc_left
            Fint_m1e[1] += GP_weights_scaled[iGP] * fC.analyticGaussianSolution(xGP, 0, sigma, wavespeed, -deltaT) * BasisFunc_right

            # Element mass matrix
            Me[0, 0] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_left ** 2
            Me[0, 1] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_left * BasisFunc_right
            Me[1, 1] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_right ** 2

            # Element stiffness matrix
            Ke[0, 0] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_left ** 2
            Ke[0, 1] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_left * DerBasisFunc_right
            Ke[1, 1] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_right ** 2

        # Assembly element matrices because of symmetry
        Me[1,0] = Me[0,1]
        Ke[1, 0] = Ke[0, 1]

        # Scatter to global matrices
        Mg[iEl:iEl+2, iEl:iEl+2] += Me * detJacobian
        Kg[iEl:iEl+2, iEl:iEl+2] += Ke * detJacobian
        Fint_0[iEl:iEl+2] += Fint_0e * detJacobian
        Fint_m1[iEl:iEl+2] += Fint_m1e * detJacobian

    # Convert to CSR format for efficient operations
    Mg = Mg.tocsr()
    Kg = Kg.tocsr()

    # Use sparse solver to compute uhat0 and uhatm1
    uhat0 = spsolve(Mg, Fint_0)
    uhatm1 = spsolve(Mg, Fint_m1)


    uhat = np.zeros((n, nTimesteps))
    uhat[:, 0] = np.squeeze(uhat0)
    uhat[:, 1] = 2 * uhat0 - uhatm1 - deltaT**2 * spsolve(Mg, Kg @ uhat0)

    # Time stepping with Central Difference Method (CDM)
    T = 0
    for i in range(2, nTimesteps):
        uhat[:, i] = 2 * uhat[:, i-1] - uhat[:, i-2] - deltaT**2 * spsolve(Mg, Kg @ uhat[:, i-1])
        T += deltaT

    # end time
    end = time.time()

    # compute run time
    time_s = end - start

    # compute L2 error for time T
    x_Points = nodes
    error = np.zeros(n)
    ref = np.zeros(n)
    for i_X in range(len(x_Points)):

        # analytical solution for time T and same initial displacment
        ref_uhat = fC.analyticGaussianSolution(x_Points[i_X], 0, sigma, wavespeed, T)

        error[i_X] = (uhat[i_X, -1] - ref_uhat) ** 2
        ref[i_X] = ref_uhat ** 2

    L2 = np.sqrt( trapezoid(error, x_Points, dx=delta_x[i_El]) / trapezoid(ref, x_Points, dx=delta_x[i_El]) )

    # save results
    #np.save(f"FEM_{n-1}_time", time_s)
    #np.save(f"FEM_{n-1}_L2", L2)
    #np.save(f"FEM_{n-1}_X", x_Points)

    # print results for pre-checking
    #print(T)
    #print(time_s)
    #print(L2)