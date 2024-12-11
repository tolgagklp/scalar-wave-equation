import numpy as np
import os
import matplotlib.pyplot as plt

def import_L2(method, elements):
    script = os.path.dirname(os.path.abspath("Conv_Plot.py"))
    folder = f"Conv_FEM_Space"
    file_name = f"{method}_{elements}_L2.npy"

    return np.load(os.path.join(script, folder, file_name))

def import_time(method, elements):
    script = os.path.dirname(os.path.abspath("Conv_Plot.py"))
    folder = f"Conv_FEM_Space"
    file_name = f"{method}_{elements}_time.npy"

    return np.load(os.path.join(script, folder, file_name))

# Elements and delta_x
Elements = [80, 160, 320, 640, 1280]
delta_x = [16/80, 16/160, 16/320, 16/640, 16/1280]

# initializing arrays
FDM = []
FEM = []
time_FDM = []
time_FEM = []
# import results for all dx / element sizes
for i_El in range(len(Elements)):
    FDM.append(import_L2("FDM", Elements[i_El]))
    FEM.append(import_L2("FEM", Elements[i_El]))
    time_FDM.append(import_time("FDM", Elements[i_El]))
    time_FEM.append(import_time("FEM", Elements[i_El]))

# reference line
ref_error = [0.7 * FDM[-1]]
k = ref_error[0] / delta_x[-1]**2
ref_error.append(k * delta_x[-2]**2)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Two rows, one column

# First plot: L2 error
axs[0].plot(delta_x, FDM, lw=1.75, color="red", ls="-", marker="s", ms=5, label="FDM")
axs[0].plot(delta_x, FEM, lw=1.75, color="blue", ls="--", marker="o", ms=5, label="FEM")
axs[0].plot(delta_x[-1:-3:-1], ref_error, color="lime")

# Plot text
x_text = 9 * (16/640) / 12
y_text = 2.5e-4
axs[0].text(x=x_text, y=y_text, s="$\propto h^2$", fontsize=14, color='black')

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].tick_params(axis='x', direction='in', which="both", labelsize=14)
axs[0].tick_params(axis='y', direction='in', which="both", labelsize=14)
axs[0].set_xlabel("Spatial discretization $dx = h$", fontsize=16)
axs[0].set_ylabel("$L_{2}$ error", fontsize=16)
axs[0].grid(True, alpha=0.5, which="major", linestyle="--")
axs[0].legend(loc="upper left", fancybox=False, fontsize=16)

# Second plot: Runtime
axs[1].plot(delta_x, time_FDM, lw=1.75, color="red", ls="-", marker="s", ms=5, label="FDM")
axs[1].plot(delta_x, time_FEM, lw=1.75, color="blue", ls="--", marker="o", ms=5, label="FEM")
axs[1].set_xscale("log")
axs[1].tick_params(axis='x', direction='in', which="both", labelsize=14)
axs[1].tick_params(axis='y', direction='in', which="both", labelsize=14)
axs[1].set_xlabel("Spatial discretization $dx=h$", fontsize=16)
axs[1].set_ylabel("Runtime $[s]$", fontsize=16)
axs[1].grid(True, alpha=0.5, which="major", linestyle="--")
axs[1].legend(loc="upper right", fancybox=False, fontsize=16)


# Adjust layout
#plt.tight_layout()
plt.show()
