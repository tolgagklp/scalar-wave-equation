import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes import System

# Material properties
wave_speed = 1.0  # Wave propagation speed (m/s)

# Time-stepping parameters
total_time = 2.0       # Total simulation time (s)
time_steps = 100       # Number of time steps

# Initialize the system
system = System(filename="data")

system.get_info()

middle_node_index = ( system.num_nodes + 1 ) // 2

# Plot the displacement time series for a sample node (e.g., Node 10)
plt.plot(np.linspace(0, total_time, time_steps), system.u_solved[middle_node_index, :])
plt.xlabel("Time (s)")
plt.ylabel(f"Displacement at Node {middle_node_index}")
plt.title("Time Series of Displacement at a Specific Node")
plt.show()

# Animate wave propagation over time
system.animate_wave_propagation_time(total_time, time_steps)
