from classes import System
import numpy as np
"""
PLEASE CHANGE GAMMA FUNCTION IN classes.py TO DEFINE DAMAGED DOMAIN.
"""

# Material properties
wave_speed = 5.0  # Wave propagation speed (m/s)
domain_x = [-1.0, 1.0]  # Domain boundaries along the X-axis
domain_y = [-1.0, 1.0]  # Domain boundaries along the Y-axis
num_elements_x = 50    # Number of elements along the X-axis
num_elements_y = 50  # Number of elements along the Y-axis
total_time = 2       # Total simulation time (s)  

# Time-stepping parameter
time_steps = 2500     # Number of time steps

# Initialize the system
system = System(domain_x, domain_y, num_elements_x, num_elements_y, wave_speed)

# Solve the system
system.solve_with_initial_conditions(total_time, time_steps)

# Animate the wave solution
anim = system.animate_wave_solution(time_steps)


