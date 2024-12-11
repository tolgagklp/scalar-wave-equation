import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.legendre as geek
import matplotlib.animation as animation
import func
from matplotlib.animation import PillowWriter

# material parameters
density = 1.0
wavespeed = 1.0

# initial displacement
frequency = 1.0
lambda_dom = wavespeed / frequency
sigma = lambda_dom / 2 / np.pi

# computational domain
xMin = -2.0
xMax = 2.0
# interface to fictitious domain
interface = 2.0
alphaFCM = 1e-5

# specifying discretization in space
n = 161
nodes = np.linspace(xMin, xMax, n)

# specifying Time stepping
deltaT = 1e-3
Tmax = 2*(2 * interface) / wavespeed
nTimesteps = int(Tmax / deltaT) + 1

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

# vectors for projecting the initial conditions
Fint_0 = np.zeros((n, 1))
Fint_m1 = np.zeros((n, 1))

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
    Fint_0e = np.zeros((2, 1))
    Fint_m1e = np.zeros((2, 1))

    # Gauss points mapped to element
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

        # Mass and stiffness matrices
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
    Fint_0[iEl:iEl+2] += Fint_0e
    Fint_m1[iEl:iEl+2] += Fint_m1e

# Compute inverse global mass matrix for the time stepping
invM = np.linalg.inv(Mg)

# Projection of initial conditions onto discretization basis
uhat0 = invM @ Fint_0
uhatm1 = invM @ Fint_m1

t = np.linspace(0.0, Tmax, nTimesteps)
uhat = np.zeros((n, nTimesteps))
uhat[:, 0] = np.squeeze(uhat0)
uhat[:, 1] = np.squeeze(2 * uhat0 - uhatm1 - deltaT**2 * (invM @ (Kg @ uhat0)))
Fg = np.zeros((n, nTimesteps))
Fg[int(n/2),:] = func.generate_SinBurst(frequency, 1, 0.0005 ,t)


# Time stepping with Central Difference Method (CDM)
for i in range(2, nTimesteps):
    uhat[:, i] = 2 * uhat[:, i-1] - uhat[:, i-2] - deltaT**2 * (invM @ (Kg @ uhat[:, i-1]))  + Fg[:,i]


# Plotting initial solution
nPoints = n
xPlot = np.linspace(xMin, xMax, nPoints)
usim = np.zeros(nPoints)
for j, x in enumerate(xPlot):
    iEl = min(n - 2, int((x - xMin) / (xMax - xMin) * (n - 1)))
    x1, x2 = nodes[iEl], nodes[iEl + 1]
    if x1 <= x <= x2:
        usim[j] = uhat[iEl, 0] * basis_function_left(x, x1, x2) + uhat[iEl + 1, 0] * basis_function_right(x, x1, x2)

fig, ax = plt.subplots()
line1, = ax.plot(xPlot, usim, ls="-")

# Animation of the solution
def animate(iT):
    usim = np.zeros(nPoints)
    for j, x in enumerate(xPlot):
        iEl = min(n - 2, int((x - xMin) / (xMax - xMin) * (n - 1)))
        x1, x2 = nodes[iEl], nodes[iEl + 1]
        if x1 <= x <= x2:
            usim[j] = uhat[iEl, iT] * basis_function_left(x, x1, x2) + uhat[iEl + 1, iT] * basis_function_right(x, x1, x2)
    line1.set_ydata(usim)
    return line1

ani = animation.FuncAnimation(fig, animate, range(0, nTimesteps, 50), interval=175, repeat=True)
# Plot parameters
plt.ylim([-0.5, 2.2])
plt.xlim([xMin, xMax])
plt.tick_params(axis='x', direction='in', which="both", labelsize=13)
plt.tick_params(axis='y', direction='in', which="both", labelsize=13)
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$u(x)$", fontsize=16)
plt.grid(True, alpha=0.5, which="major", linestyle="--")
plt.show()

#writer = PillowWriter(fps=10)
#ani.save("1D_FEM_Damage.gif", writer=writer)


