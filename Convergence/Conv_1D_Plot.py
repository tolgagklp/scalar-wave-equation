import numpy as np
import os
import matplotlib.pyplot as plt

# Function to import L2 error values from saved files
def import_L2(method, elements):
    """
    Import the L2 error for a specific method and number of elements.

    Parameters:
        method (str): The method name (e.g., "FDM" or "FEM").
        elements (int): The number of elements / dx used in the discretization.

    Returns:
        np.ndarray: Array of L2 error values.
    """

    script = os.path.dirname(os.path.abspath("Conv_1D_Plot.py"))
    folder = f"Conv_FEM_Space"
    file_name = f"{method}_{elements}_L2.npy"

    return np.load(os.path.join(script, folder, file_name))

# Function to import runtime values from saved files
def import_time(method, elements):
    """
    Import the runtime for a specific method and number of elements.

    Parameters:
        method (str): The method name (e.g., "FDM" or "FEM").
        elements (int): The number of elements / dx used in the discretization.

    Returns:
        np.ndarray: Array of runtime values.
    """

    script = os.path.dirname(os.path.abspath("Conv_1D_Plot.py"))
    folder = f"Conv_FEM_Space"
    file_name = f"{method}_{elements}_time.npy"

    return np.load(os.path.join(script, folder, file_name))

# Elements and spatial discretization sizes
Elements = [80, 160, 320, 640, 1280]
delta_x = [16/80, 16/160, 16/320, 16/640, 16/1280]

# Initializing arrays to store L2 error and runtime values
FDM = []
FEM = []
time_FDM = []
time_FEM = []

# Import results for all element sizes (delta_x values)
for i_El in range(len(Elements)):
    FDM.append(import_L2("FDM", Elements[i_El]))
    FEM.append(import_L2("FEM", Elements[i_El]))
    time_FDM.append(import_time("FDM", Elements[i_El]))
    time_FEM.append(import_time("FEM", Elements[i_El]))

# Calculate a reference error for plotting convergence lines
ref_error = [0.7 * FDM[-1]]
k = ref_error[0] / delta_x[-1]**2
ref_error.append(k * delta_x[-2]**2)

# Create subplots for L2 error and runtime
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Two plots side by side

# First plot: L2 error
axs[0].plot(delta_x, FDM, lw=1.75, color="red", ls="-", marker="s", ms=5, label="FDM")
axs[0].plot(delta_x, FEM, lw=1.75, color="blue", ls="--", marker="o", ms=5, label="FEM")
axs[0].plot(delta_x[-1:-3:-1], ref_error, color="lime")

# Annotate the reference convergence rate
x_text = 9 * (16/640) / 12
y_text = 2.5e-4
axs[0].text(x=x_text, y=y_text, s="$\propto h^2$", fontsize=14, color='black')

# Set logarithmic scales and labels for the first plot
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

# Set logarithmic scales and labels for the second plot
axs[1].set_xscale("log")
axs[1].tick_params(axis='x', direction='in', which="both", labelsize=14)
axs[1].tick_params(axis='y', direction='in', which="both", labelsize=14)
axs[1].set_xlabel("Spatial discretization $dx=h$", fontsize=16)
axs[1].set_ylabel("Runtime $[s]$", fontsize=16)
axs[1].grid(True, alpha=0.5, which="major", linestyle="--")
axs[1].legend(loc="upper right", fancybox=False, fontsize=16)

# Show the plots
plt.show()
