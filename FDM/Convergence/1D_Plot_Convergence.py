import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def import_numerical(xPoints, sigma, timeSteps):
    script = os.path.dirname(os.path.abspath("Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"u_FDM_xPoints_{xPoints}_dt_0.001_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_analytical(xPoints, sigma, timeSteps):
    script = os.path.dirname(os.path.abspath("Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"u_ref_xPoints_{xPoints}_dt_0.001_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_xPoints(xPoints, sigma):
    script = os.path.dirname(os.path.abspath("Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"x_xPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))

xMin = 0
xMax = 10
#xPoints = [101, 251, 501, 1001, 2501, 5001]
xPoints = [51, 101, 201, 401, 601]

timeSteps = 2500
sigma = str(0.5)

L2 = np.zeros(len(xPoints))
delta_x = np.zeros(len(xPoints))

for i in range(len(xPoints)):

    x = import_xPoints(xPoints[i], sigma)
    u_num = import_numerical(xPoints[i], sigma, timeSteps)
    u_ref = import_analytical(xPoints[i], sigma, timeSteps)
    delta_x[i] = 10 / (xPoints[i] - 1)

    #plt.plot(x, u_num, "r")
    #plt.plot(x, u_ref, "b")
    #plt.show()


    error = np.zeros(len(x))
    ref = np.zeros(len(x))

    for j in range(len(x)):
        error[j] = np.power((u_num[j]- u_ref[j]), 2)
        ref[j] = np.power(u_ref[j], 2)

    L2[i] = np.sqrt(trapezoid(error, x, dx=delta_x[i]) / trapezoid(ref, x, dx=delta_x[i]))


plt.plot(delta_x, L2, color="orange", lw=2.0, ls="-", marker="s", ms=5, label="$2500$ steps of dt$=0.001$")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-4, 1e-1)
plt.tick_params(axis='x', direction='in', which="both")
plt.tick_params(axis='y', direction='in', which="both")

plt.xlabel("dx", fontsize=14)
plt.ylabel("$L_{2}$ error", fontsize=14)
plt.grid(True, alpha=0.5, which="major", linestyle="--")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()
