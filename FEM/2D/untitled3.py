import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classes import System
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', depending on your setup
plt.ion()  # Turn on interactive mode

# Material properties
wavespeed = 1.0

# Mesh inputs
domain_x = [-2, 2]
domain_y = [-2, 2]
num_elements_x = 5
num_elements_y = 5
num_frequencies = 50
frequencies = np.linspace(0, 100, num_frequencies)

# Initialize the system with mesh parameters
system = System(domain_x, domain_y, num_elements_x, num_elements_y, wavespeed)

# After creating the system
system.apply_boundary_conditions()  # Ensure boundary conditions are applied

# Solve the system in the frequency domain
displacement = system.solve_frequency_domain(frequencies)

total_time = 2.0  # Total time for simulation
time_steps = 200  # Number of time steps
# Transform the frequency domain solution to the time domain
F = system.force_vector
u_time = 1000 * system.transform_to_time_domain(frequencies, displacement, total_time, time_steps)

# Animate the wave propagation in the time domain
system.animate_wave_propagation_time(u_time, total_time, time_steps)
