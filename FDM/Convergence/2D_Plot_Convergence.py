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

timeSteps = 1400
dt = 0.0005

delta = np.zeros((len(nPoints)))
L2 = np.zeros((len(nPoints)))

for k in range(len(nPoints)):

    x = import_xPoints(nPoints[k])
    y = import_yPoints(nPoints[k])
    u_num = import_numerical(nPoints[k], timeSteps, dt)
    u_ref = import_analytical(nPoints[k], timeSteps, dt)

    delta[k] = 1 / (nPoints[k] - 1)

    error = np.zeros((nPoints[k], nPoints[k]))
    ref = np.zeros((nPoints[k], nPoints[k]))

    for i in range(nPoints[k]):
        for j in range(nPoints[k]):
            error[i,j] = (u_num[i,j] - u_ref[i,j]) ** 2
            ref[i,j] = u_ref[i,j] ** 2

    error_int_y = trapezoid(error, y, dx=delta[k], axis=0)
    error_int = trapezoid(error_int_y, x, dx=delta[k])

    ref_int_y = trapezoid(ref, y, dx=delta[k], axis=0)
    ref_int = trapezoid(ref_int_y, x, dx=delta[k])

    L2[k] = np.sqrt(error_int) / np.sqrt(ref_int)


plt.plot(delta, L2, lw=1.75, color="red", ls="-", marker="s", ms=5, label="$1400$ steps with dt$=0.0005$")
plt.xscale("log")
plt.yscale("log")
plt.tick_params(axis='x', direction='in', which="both")
plt.tick_params(axis='y', direction='in', which="both")

plt.xlabel("dx, dy", fontsize=14)
plt.ylabel("$L_{2}$ error", fontsize=14)
plt.grid(True, alpha=0.5, which="major", linestyle="--")
plt.tight_layout()
plt.legend(loc="upper left", fancybox=False)
plt.show()