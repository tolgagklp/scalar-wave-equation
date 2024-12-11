import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def import_numerical(iPoints, timeSteps, dt):
    """
    Imports numerical solution data from a file.

    Parameters:
        iPoints (int): Number of points in the spatial discretization.
        timeSteps (int): Number of time steps in the simulation.
        dt (float): Time step size.

    Returns:
        numpy.ndarray: Numerical solution array loaded from the specified file.
    """
    script = os.path.dirname(os.path.abspath("Conv_2D_Plot.py"))
    folder = "Data"
    file_name = f"u_FDM_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))

def import_analytical(iPoints, timeSteps, dt):
    """
    Imports analytical solution data from a file.

    Parameters:
        iPoints (int): Number of points in the spatial discretization.
        timeSteps (int): Number of time steps in the simulation.
        dt (float): Time step size.

    Returns:
        numpy.ndarray: Analytical solution array loaded from the specified file.
    """
    script = os.path.dirname(os.path.abspath("Conv_2D_Plot.py"))
    folder = "Data"
    file_name = f"u_ref_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))

def import_xPoints(xPoints):
    """
    Imports x-coordinate points from a file.

    Parameters:
        xPoints (int): Number of points in the x-direction.

    Returns:
        numpy.ndarray: Array of x-coordinate points loaded from the specified file.
    """
    script = os.path.dirname(os.path.abspath("Conv_2D_Plot.py"))
    folder = "Data"
    file_name = f"xPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))

def import_yPoints(xPoints):
    """
    Imports y-coordinate points from a file.

    Parameters:
        xPoints (int): Number of points in the y-direction (assumed equal to xPoints).

    Returns:
        numpy.ndarray: Array of y-coordinate points loaded from the specified file.
    """
    script = os.path.dirname(os.path.abspath("Conv_2D_Plot.py"))
    folder = "Data"
    file_name = f"yPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))

# domain
Min = 0
Max = 1
nPoints = [11, 21, 41, 81, 161]

# time
timeSteps = 1400
dt = 0.0005

# initialize arrays
delta = np.zeros((len(nPoints)))
L2 = np.zeros((len(nPoints)))
for k in range(len(nPoints)):

    # import results
    x = import_xPoints(nPoints[k])
    y = import_yPoints(nPoints[k])
    u_num = import_numerical(nPoints[k], timeSteps, dt)
    u_ref = import_analytical(nPoints[k], timeSteps, dt)

    # compute dx=dy for integration
    delta[k] = 1 / (nPoints[k] - 1)

    # initialize arrays
    error = np.zeros((nPoints[k], nPoints[k]))
    ref = np.zeros((nPoints[k], nPoints[k]))
    for i in range(nPoints[k]):
        for j in range(nPoints[k]):
            error[i,j] = (u_num[i,j] - u_ref[i,j]) ** 2
            ref[i,j] = u_ref[i,j] ** 2

    # integration over both directions for error
    error_int_y = trapezoid(error, y, dx=delta[k], axis=0)
    error_int = trapezoid(error_int_y, x, dx=delta[k])

    # integration over both directions for reference solution
    ref_int_y = trapezoid(ref, y, dx=delta[k], axis=0)
    ref_int = trapezoid(ref_int_y, x, dx=delta[k])

    # L2 error
    L2[k] = np.sqrt(error_int) / np.sqrt(ref_int)

# plot results
plt.plot(delta, L2, lw=1.75, color="red", ls="-", marker="s", ms=5, label="$1400$ steps with dt$=0.0005$")
# plot parameters
plt.xscale("log")
plt.yscale("log")
plt.tick_params(axis='x', direction='in', which="both")
plt.tick_params(axis='y', direction='in', which="both")
plt.xlabel("dx, dy", fontsize=14)
plt.ylabel("$L_{2}$ error", fontsize=14)
plt.grid(True, alpha=0.5, which="major", linestyle="--")
plt.legend(loc="upper left", fancybox=False)

plt.show()
