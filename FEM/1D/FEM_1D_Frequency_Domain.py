import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import numpy.polynomial.legendre as geek
import FEMfunc as func

# material parameters
density = 1.0
wavespeed = 1.0

# computational domain
xMin = -1.0
xMax = 1.0

# specifying discretization in space
n = 100
nodes = np.linspace(xMin, xMax, n)


# linear basis functions and their derivatives
def basis_function_left(x, node1, node2):
    return (node2 - x) / (node2 - node1)

def basis_function_right(x, node1, node2):
    return (x - node1) / (node2 - node1)

def basis_function_left_derivative(node1, node2):
    return -1.0 / (node2 - node1)

def basis_function_right_derivative(node1, node2):
    return 1.0 / (node2 - node1)

# initializing global system matrices
Mg = np.zeros((n, n))
Kg = np.zeros((n, n))

# integration rule
nGP = 2
GP = geek.leggauss(nGP)[0]
GP_weights = geek.leggauss(nGP)[1]

nEl = n - 1
# loop over all elements
for iEl in range(0, nEl):
    # Element nodes
    node1 = nodes[iEl]
    node2 = nodes[iEl+1]

    # Element matrices and load vectors
    Me = np.zeros((2, 2))
    Ke = np.zeros((2, 2))

    # Gauss points mapped to element
    GP_global = [node1 + 0.5 * (node2 - node1) * (gp + 1) for gp in GP]
    GP_weights_scaled = GP_weights * (node2 - node1) / 2.0

    # loop over Gauss points
    for iGP, xGP in enumerate(GP_global):
        alphaGP = 1

        # Evaluate basis functions and their derivatives at xGP
        BasisFunc_left = basis_function_left(xGP, node1, node2)
        BasisFunc_right = basis_function_right(xGP, node1, node2)
        DerBasisFunc_left = basis_function_left_derivative(node1, node2)
        DerBasisFunc_right = basis_function_right_derivative(node1, node2)

        # Mass and stiffness matricesDerBasisFunc
        Me[0, 0] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_left ** 2
        Me[0, 1] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_left * BasisFunc_right
        Me[1, 1] += alphaGP * density * GP_weights_scaled[iGP] * BasisFunc_right ** 2

        Ke[0, 0] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_left ** 2
        Ke[0, 1] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_left * DerBasisFunc_right
        Ke[1, 1] += alphaGP * density * wavespeed ** 2 * GP_weights_scaled[iGP] * DerBasisFunc_right ** 2

    Me[1, 0] = Me[0, 1]
    Ke[1, 0] = Ke[0, 1]

    # Scatter to global matrices
    Mg[iEl:iEl+2, iEl:iEl+2] += Me
    Kg[iEl:iEl+2, iEl:iEl+2] += Ke


# Parameters for solving in frequency and time domain
N = 499999
dt = 0.025
Nx = n
rho = density
vs = wavespeed

mass_matrix = Mg[1:-1, 1:-1]
stiff_matrix = Kg[1:-1, 1:-1]

# Generate force
# Apply the force at node 49
frequency = 0.1
cycles = 3
amplitude = 1
t = np.linspace(0, dt * (N + 1), N + 1)
Force = func.generateSineBurst(frequency, cycles, amplitude, t, Nx, N)
plt.plot(Force[49, :5000])

# Time domain solution using newmark's method
u_displacement_FEM, _, _ = func.newmark(M=mass_matrix, K=stiff_matrix, C=np.zeros_like(mass_matrix), F=Force, u0=np.zeros_like(Force.shape[0]), ut0=np.zeros_like(Force.shape[0]), nt=N, dt=dt)

# Frequency domain solution
freq_axis = np.fft.rfftfreq(len(t), d=dt)

# Calculate the fourier transform of the force function
force_freq = np.zeros((Nx - 2, len(freq_axis)))
force_freq = np.fft.rfft((Force))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
index = 3000
ax1.plot(freq_axis[:index], np.abs(force_freq[49, :index]))
# Phase plot
ax2.plot(freq_axis[:index], np.unwrap(np.angle(force_freq[49, :index])))

# Solving the system in the frequency domain
u_freq = np.zeros((mass_matrix.shape[0], len(freq_axis)), dtype=complex)
A = np.zeros((mass_matrix.shape[0], mass_matrix.shape[1]), dtype=complex)

# Frequency loop up to 1Hz
index = 3000
for i in range(index):
    if i == 0:
        continue
    A = (stiff_matrix - (2 * np.pi * freq_axis[i]) ** 2 * mass_matrix)
    A_csc = csc_matrix(A)
    u_freq[:, i] = scipy.sparse.linalg.spsolve(A_csc, (force_freq[:, i]))
    if i % 10 == 0:
        print(f"Solving frequency step: {i} / {index}")

# print(u_freq.shape)
u_freq = np.reshape(u_freq, ((Nx - 2), -1))
u_time2freq = np.fft.rfft(u_displacement_FEM, axis=-1)

# Compare time domain solution with frequency domain solution
u_freq2time = np.fft.irfft(u_freq, len(t))

#Index for steady state
index_steadystate = 300000
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5))
ax1.plot(t[index_steadystate:index_steadystate + 10000],(u_freq2time[15, index_steadystate:index_steadystate + 10000].real), lw=1.75, linestyle="-", label="Solved in frequency domain")
ax1.plot(t[index_steadystate:index_steadystate + 10000],u_displacement_FEM[15, index_steadystate:index_steadystate + 10000], lw=1.75, linestyle="--", label="Solved in time domain")

ax1.set_xlabel("Time $[s]$", fontsize=16)
ax1.set_ylabel("Displacement $u$", fontsize=16)
ax1.set_xlim([7500, 7750])
ax1.set_ylim([-0.25, 0.35])

ax1.tick_params(axis='x', direction='in', which="both", labelsize=13)
ax1.tick_params(axis='y', direction='in', which="both", labelsize=13)
ax1.grid(True, alpha=0.5, which="major", linestyle="--")
ax1.legend(loc="upper right", fancybox=False, fontsize=16)
plt.tight_layout()
plt.show()