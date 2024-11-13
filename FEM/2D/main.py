import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes import System

# Material properties
wave_speed = 1.0  # Wave propagation speed (m/s)

# Mesh inputs (domain boundaries and element counts)
domain_x = [-10.0, 10.0]  # Domain boundaries along the X-axis
domain_y = [-10.0, 10.0]  # Domain boundaries along the Y-axis
num_elements_x = 120    # Number of elements along the X-axis
num_elements_y = 120    # Number of elements along the Y-axis

# Time-stepping parameters
total_time = 2.0        # Total simulation time (s)
time_steps = 100        # Number of time steps

# Initialize the system
system = System(domain_x, domain_y, num_elements_x, num_elements_y, wave_speed)

# Solve the system in the time domain
u_time = system.solve_time_domain(total_time, time_steps, "data")

system.get_info(detailed=True)


# Plot the displacement time series for a sample node
middle_node_index = ( system.num_nodes + 1 ) // 2

plt.plot(np.linspace(0, total_time, time_steps), system.u_solved[middle_node_index, :])
plt.xlabel("Time (s)")
plt.ylabel(f"Displacement at Node {middle_node_index}")
plt.title("Time Series of Displacement at a Specific Node")
plt.show()

# Animate wave propagation over time
system.animate_wave_propagation_time(total_time, time_steps)
