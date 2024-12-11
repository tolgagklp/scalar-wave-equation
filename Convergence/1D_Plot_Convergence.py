import numpy as np
import os
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def import_numerical(xPoints, sigma, timeSteps, dt):
    script = os.path.dirname(os.path.abspath("1D_Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"u_FDM_xPoints_{xPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_analytical(xPoints, sigma, timeSteps, dt):
    script = os.path.dirname(os.path.abspath("1D_Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"u_ref_xPoints_{xPoints}_dt_{dt}_timeSteps_{timeSteps}.npy"

    return np.load(os.path.join(script, folder, file_name))
def import_xPoints(xPoints, sigma):
    script = os.path.dirname(os.path.abspath("1D_Plot_Convergence.py"))
    folder = f"Data_sigma_{sigma}"
    file_name = f"x_xPoints_{xPoints}.npy"

    return np.load(os.path.join(script, folder, file_name))

xMin = 0
xMax = 16
#xPoints = [101, 251, 501, 1001, 2501, 5001]
#xPoints = [51, 101, 201, 401, 601, 1201]
xPoints = [81, 161, 321, 641, 1281]


timeSteps = 10000
dt = 0.0005
sigma = str(0.5)

L2 = np.zeros(len(xPoints))
delta_x = np.zeros(len(xPoints))

for i in range(len(xPoints)):

    x = import_xPoints(xPoints[i], sigma)
    u_num = import_numerical(xPoints[i], sigma, timeSteps, dt)
    u_ref = import_analytical(xPoints[i], sigma, timeSteps, dt)
    delta_x[i] = 16 / (xPoints[i] - 1)

    '''
    plt.plot(x, u_num, linestyle="-", linewidth=1.8, label="$u$(x,$T_{max}$)")
    plt.plot(x, u_ref, linestyle="--", linewidth=1.8, label="$u_{ref}$(x,$T_{max}$)")
    plt.xlim(0, 16)
    plt.ylim(-0.05, 0.7)
    plt.tick_params(axis='x', direction='in', which="both")
    plt.tick_params(axis='y', direction='in', which="both")
    plt.legend(loc="upper left", fancybox=False)
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.show()
    '''

    error = np.zeros(len(x))
    ref = np.zeros(len(x))

    for j in range(len(x)):
        error[j] = (u_num[j]- u_ref[j]) ** 2
        ref[j] = u_ref[j] ** 2

    L2[i] = np.sqrt(trapezoid(error, x, dx=delta_x[i]) / trapezoid(ref, x, dx=delta_x[i]))


plt.plot(delta_x, L2, lw=1.75, color="red", ls="-", marker="s", ms=5, label=f"${timeSteps}$ steps with dt$={dt}$")
plt.xscale("log")
plt.yscale("log")
#plt.ylim(1e-4, 1e-1)
plt.tick_params(axis='x', direction='in', which="both")
plt.tick_params(axis='y', direction='in', which="both")

plt.xlabel("dx", fontsize=14)
plt.ylabel("$L_{2}$ error", fontsize=14)
plt.grid(True, alpha=0.5, which="major", linestyle="--")
plt.tight_layout()
plt.legend(loc="upper left", fancybox=False)
plt.show()
