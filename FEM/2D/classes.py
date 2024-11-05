import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Gamma function for spatially varying density
def gamma_function(x, y):
    return 1

class Node:
    def __init__(self, x, y, u=0):
        self.x = x
        self.y = y
        self.u = u
        self.shape_xi = None
        self.shape_eta = None

class Element:
    def __init__(self, node1, node2, node3, node4, wavespeed):
        self.nodes = [node1, node2, node3, node4]
        self.c = wavespeed
        self.stiff = np.zeros((4, 4))
        self.mass = np.zeros((4, 4))
        self.set_isoparametric_coordinates()
        self.stiff, self.mass = self.calculate_matrices()

    def set_isoparametric_coordinates(self):
        self.nodes[0].shape_xi, self.nodes[0].shape_eta = -1, -1  # Bottom-left
        self.nodes[1].shape_xi, self.nodes[1].shape_eta = 1, -1   # Bottom-right
        self.nodes[2].shape_xi, self.nodes[2].shape_eta = 1, 1    # Top-right
        self.nodes[3].shape_xi, self.nodes[3].shape_eta = -1, 1   # Top-left

    def shape_function_derivatives(self, xi, eta):
        dN_dxi = [-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4]
        dN_deta = [-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4]
        return dN_dxi, dN_deta

    def jacobian(self, dN_dxi, dN_deta):
        J = np.zeros((2, 2))
        for i, node in enumerate(self.nodes):
            J[0, 0] += dN_dxi[i] * node.x
            J[0, 1] += dN_dxi[i] * node.y
            J[1, 0] += dN_deta[i] * node.x
            J[1, 1] += dN_deta[i] * node.y
        return J, np.linalg.det(J), np.linalg.inv(J)

    def calculate_B_matrix(self, invJ, dN_dxi, dN_deta):
        B = np.zeros((2, 4))
        for i in range(4):
            dN_dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
            dN_dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
            B[0, i] = dN_dx
            B[1, i] = dN_dy
        return B

    def calculate_matrices(self):
        K = np.zeros((4, 4))
        M = np.zeros((4, 4))
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        weights = [1, 1]

        for xi in gauss_points:
            for eta in gauss_points:
                N = [
                    0.25 * (1 - xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 + eta),
                    0.25 * (1 - xi) * (1 + eta)
                ]
                
                dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
                J, detJ, invJ = self.jacobian(dN_dxi, dN_deta)
                B = self.calculate_B_matrix(invJ, dN_dxi, dN_deta)
                
                x = sum(node.x * N[i] for i, node in enumerate(self.nodes))
                y = sum(node.y * N[i] for i, node in enumerate(self.nodes))
                gamma = gamma_function(x, y)
                
                for i in range(4):
                    for j in range(4):
                        K[i, j] += gamma * self.c**2 * (B[:, i] @ B[:, j]) * detJ * weights[0] * weights[1]
                        M[i, j] += gamma * N[i] * N[j] * detJ * weights[0] * weights[1]

        return K, M

def create_mesh(domain_x, domain_y, num_elements_x, num_elements_y, wavespeed):
    x_positions = np.linspace(domain_x[0], domain_x[1], num_elements_x + 1)
    y_positions = np.linspace(domain_y[0], domain_y[1], num_elements_y + 1)
    nodes = [Node(x, y) for y in y_positions for x in x_positions]

    elements = []
    for j in range(num_elements_y):
        for i in range(num_elements_x):
            node1 = nodes[j * (num_elements_x + 1) + i]
            node2 = nodes[j * (num_elements_x + 1) + (i + 1)]
            node3 = nodes[(j + 1) * (num_elements_x + 1) + (i + 1)]
            node4 = nodes[(j + 1) * (num_elements_x + 1) + i]
            
            elements.append(Element(node1, node2, node3, node4, wavespeed))
    
    return nodes, elements

class System:
    def __init__(self, domain_x, domain_y, num_elements_x, num_elements_y, wavespeed):
        self.domain_x = domain_x
        self.domain_y = domain_y

        self.nodes, self.elements = create_mesh(domain_x, domain_y, num_elements_x, num_elements_y, wavespeed)
        self.num_nodes = len(self.nodes)
        self.global_stiffness = np.zeros((self.num_nodes, self.num_nodes))
        self.global_mass = np.zeros((self.num_nodes, self.num_nodes))
        #self.force_vector = np.zeros(self.num_nodes)
        self.force_vector = np.zeros(self.num_nodes)

        # Identify boundary nodes by category
        self.left_boundary, self.right_boundary, self.top_boundary, self.bottom_boundary = self.identify_boundary_nodes()

        # Assemblying global matrices
        self.assemble_global_matrices()

        # Set initial conditions
        self.set_initial_conditions()

    # Storing boundary nodes
    def identify_boundary_nodes(self):
        left_boundary = []
        right_boundary = []
        top_boundary = []
        bottom_boundary = []

        for node in self.nodes:
            if node.x == self.domain_x[0]:  # Left boundary
                left_boundary.append(node)
            elif node.x == self.domain_x[1]:  # Right boundary
                right_boundary.append(node)
            if node.y == self.domain_y[1]:  # Top boundary
                top_boundary.append(node)
            elif node.y == self.domain_y[0]:  # Bottom boundary
                bottom_boundary.append(node)

        return left_boundary, right_boundary, top_boundary, bottom_boundary

    # Assemblying both stiffness and mass matrices
    def assemble_global_matrices(self):
        for element in self.elements:
            local_stiffness = element.stiff
            local_mass = element.mass
            node_indices = [self.nodes.index(node) for node in element.nodes]

            for i in range(4):
                for j in range(4):
                    global_i = node_indices[i]
                    global_j = node_indices[j]
                    self.global_stiffness[global_i, global_j] += local_stiffness[i, j]
                    self.global_mass[global_i, global_j] += local_mass[i, j]

    def apply_boundary_conditions(self):
        # For example, fixing the displacement at the boundary nodes
        for i in range(self.num_nodes):
            if (self.nodes[i].x == self.domain_x[0] or self.nodes[i].x == self.domain_x[1] or
                self.nodes[i].y == self.domain_y[0] or self.nodes[i].y == self.domain_y[1]):

                self.global_stiffness[i, :] = 0
                self.global_stiffness[i, i] = 1
                self.force_vector[i] = 0

    def set_initial_conditions(self):
        # Set initial Gaussian displacement condition
        x0 = (self.domain_x[0] + self.domain_x[1]) / 2
        y0 = (self.domain_y[0] + self.domain_y[1]) / 2
        sigma = 0.5  # Width of the Gaussian
        for node in self.nodes:
            node.u = 2 * np.exp(-((node.x - x0) ** 2 + (node.y - y0) ** 2) / (2 * sigma ** 2))

    def get_global_matrices(self):
        return self.global_stiffness, self.global_mass

    def solve_frequency_domain(self, frequencies):
        #self.set_initial_conditions()
        #self.apply_boundary_conditions()  # Ensure boundary conditions are applied before solving
        U = np.zeros((len(frequencies), self.num_nodes))
        for i, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq
            A = omega**2 * self.global_mass - self.global_stiffness
            U[i, :] = np.linalg.solve(A, self.force_vector)
        return U

    def transform_to_time_domain(self, frequencies, U, total_time, time_steps):
        dt = total_time / time_steps
        t = np.linspace(0, total_time, time_steps)

        # Initialize time-domain signal
        u_time = np.zeros((time_steps, self.num_nodes))
        u_time[1,:] = np.array([node.u for node in self.nodes])

        # Inverse Fourier Transform using numerical integration (here using a simple trapezoidal rule)
        for node_index in range(self.num_nodes):
            for time_index in range(time_steps):
                # Compute the integral for the inverse Fourier transform
                integral_sum = 0
                for freq_index, freq in enumerate(frequencies):
                    omega = 2 * np.pi * freq
                    integral_sum += U[freq_index, node_index] * np.exp(1j * omega * t[time_index])

                # Normalize by the number of frequencies (or use a more precise normalization)
                u_time[time_index, node_index] = integral_sum.real * (dt / (2 * np.pi))

        return u_time

    def animate_wave_propagation_time(self, U, total_time, time_steps):
        fig, ax = plt.subplots()
        line, = ax.plot(np.linspace(0, self.num_nodes, self.num_nodes), U[0], lw=2)

        # Set axis limits and labels
        ax.set_xlim(0, self.num_nodes - 1)
        ax.set_ylim(np.min(U) - 1, np.max(U) + 1)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Displacement')

        # Title and time display
        title_text = ax.set_title(f'Time: {0:.2f}s')
        time_text = ax.text(0.5, 0.9, '', transform=ax.transAxes)

        def update(frame):
            line.set_ydata(U[frame])  # Update the line with the new displacement
            title_text.set_text(f'Time: {frame * (total_time / time_steps):.2f}s')  # Update the title with the current time
            return line, title_text

        ani = FuncAnimation(fig, update, frames=time_steps, blit=True)
        plt.show(block=True)
