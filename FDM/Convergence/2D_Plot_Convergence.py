import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def import_numerical(iPoints, timeSteps, dt):
    script = os.path.dirname(os.path.abspath("2D_Plot_Convergence.py"))
    folder = "Data"
    file_name = f"u_FDM_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_analytical(iPoints, timeSteps, dt):
    script = os.path.dirname(os.path.abspath("2D_Plot_Convergence.py"))
    folder = "Data"
    file_name = f"u_ref_iPoints_{iPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_xPoints(xPoints):
    script = os.path.dirname(os.path.abspath("2D_Plot_Convergence.py"))
    folder = "Data"
    file_name = f"xPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_yPoints(xPoints):
    script = os.path.dirname(os.path.abspath("2D_Plot_Convergence.py"))
    folder = "Data"
    file_name = f"yPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))


Min = 0
Max = 1

nPoints = [11, 21, 41, 81, 161]

timeSteps = 700
dt = 0.001

for iPoints in nPoints:

    x = import_xPoints(iPoints)
    y = import_yPoints(iPoints)
    u_num = import_numerical(iPoints, timeSteps, dt)
    u_ref = import_analytical(iPoints, timeSteps, dt)

    delta= 1 / (iPoints - 1)

    error = np.zeros((iPoints,iPoints))
    ref = np.zeros((iPoints,iPoints))

    for i in range(iPoints):
        for j in range(iPoints):
            error[i,j] = (u_num[i,j] - u_ref[i,j]) ** 2
            ref[i,j] = u_ref[i,j] ** 2

    error_int_y = trapezoid(error, y, dx=delta, axis=0)
    error_int = trapezoid(error_int_y, x, dx=delta)

    ref_int_y = trapezoid(ref, y, dx=delta, axis=0)
    ref_int = trapezoid(ref_int_y, x, dx=delta)

    L2 = np.sqrt(error_int) / np.sqrt(ref_int)

    plt.scatter(delta, L2)

plt.xscale("log")
plt.yscale("log")
plt.show()