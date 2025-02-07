import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

# Gamma function for spatially varying density
def gamma_function(x, y):
    """
    Defines the spatially varying density function.
    
    Parameters:
    x (float): X-coordinate
    y (float): Y-coordinate
    
    Returns:
    float: Density value at the given coordinates
    """
    if -0.8 <= x <= -0.5 and -0.8 <= y <= -0.5:
        return 1.0
    return 1.0

class Node:
    """
    Represents a node in the finite element mesh.
    
    Attributes:
    x (float): X-coordinate of the node
    y (float): Y-coordinate of the node
    u (float): Displacement associated with the node (default is 0)
    shape_xi (float or None): Isoparametric coordinate in xi direction
    shape_eta (float or None): Isoparametric coordinate in eta direction
    """
    def __init__(self, x, y, u=0):
        """
        Initializes a Node object.
        
        Parameters:
        x (float): X-coordinate of the node
        y (float): Y-coordinate of the node
        u (float, optional): Value associated with the node (default is 0)
        """
        self.x = x
        self.y = y
        self.u = u
        self.shape_xi = None
        self.shape_eta = None
    
    def info(self):
        """
        Returns a string representation of the node.
        
        Returns:
        str: Node information including its coordinates
        """
        return f"Node at ({self.x}, {self.y})"

class Element:
    """
    Represents a finite element consisting of four nodes.
    
    Attributes:
    nodes (list of Node): List of four nodes defining the element
    c (float): Wave speed associated with the element
    stiff (numpy.ndarray): Stiffness matrix of the element
    mass (numpy.ndarray): Mass matrix of the element
    """
    def __init__(self, node1, node2, node3, node4, wavespeed):
        """
        Initializes an Element object.
        
        Parameters:
        node1, node2, node3, node4 (Node): Four nodes defining the element
        wavespeed (float): Wave speed associated with the element
        """
        self.nodes = [node1, node2, node3, node4]
        self.c = wavespeed
        self.stiff = np.zeros((4, 4))
        self.mass = np.zeros((4, 4))
        self.set_isoparametric_coordinates()
        self.stiff, self.mass = self.calculate_matrices()

    def set_isoparametric_coordinates(self):
        """
        Assigns isoparametric coordinates to the element's nodes.
        """
        self.nodes[0].shape_xi, self.nodes[0].shape_eta = -1, -1  # Bottom-left
        self.nodes[1].shape_xi, self.nodes[1].shape_eta = 1, -1   # Bottom-right
        self.nodes[2].shape_xi, self.nodes[2].shape_eta = 1, 1    # Top-right
        self.nodes[3].shape_xi, self.nodes[3].shape_eta = -1, 1   # Top-left

    def shape_function_derivatives(self, xi, eta):
        """
        Computes the derivatives of the shape functions with respect to xi and eta.
        
        Parameters:
        xi (float): Isoparametric coordinate in the xi direction
        eta (float): Isoparametric coordinate in the eta direction
        
        Returns:
        tuple: Two lists containing the derivatives with respect to xi and eta
        """
        dN_dxi = [-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4]
        dN_deta = [-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4]
        return dN_dxi, dN_deta

    def jacobian(self, dN_dxi, dN_deta):
        """
        Computes the Jacobian matrix for the element.

        Parameters:
        dN_dxi (list of float): Partial derivatives of shape functions with respect to xi.
        dN_deta (list of float): Partial derivatives of shape functions with respect to eta.

        Returns:
        tuple: 
            - J (numpy.ndarray): 2x2 Jacobian matrix.
            - detJ (float): Determinant of the Jacobian matrix.
            - invJ (numpy.ndarray): Inverse of the Jacobian matrix.
        """
        J = np.zeros((2, 2))
        for i, node in enumerate(self.nodes):
            J[0, 0] += dN_dxi[i] * node.x
            J[0, 1] += dN_dxi[i] * node.y
            J[1, 0] += dN_deta[i] * node.x
            J[1, 1] += dN_deta[i] * node.y
        return J, np.linalg.det(J), np.linalg.inv(J)

    def calculate_B_matrix(self, invJ, dN_dxi, dN_deta):
        """
        Computes the strain-displacement matrix (B-matrix) for the element.

        Parameters:
        invJ (numpy.ndarray): Inverse of the Jacobian matrix.
        dN_dxi (list of float): Partial derivatives of shape functions with respect to xi.
        dN_deta (list of float): Partial derivatives of shape functions with respect to eta.

        Returns:
        numpy.ndarray: 2x4 strain-displacement matrix (B-matrix).
        """
        B = np.zeros((2, 4))
        for i in range(4):
            dN_dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
            dN_dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
            B[0, i] = dN_dx
            B[1, i] = dN_dy
        return B

    def calculate_matrices(self):
        """
        Computes the stiffness and mass matrices for the element.
        
        Returns:
        tuple: Stiffness matrix and mass matrix as numpy arrays
        """
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
    """
    Creates a structured finite element mesh of quadrilateral elements.

    Parameters:
    domain_x (tuple of float): The x-axis boundaries of the domain (xmin, xmax).
    domain_y (tuple of float): The y-axis boundaries of the domain (ymin, ymax).
    num_elements_x (int): Number of elements along the x-axis.
    num_elements_y (int): Number of elements along the y-axis.
    wavespeed (float): Wave speed associated with the elements.

    Returns:
    tuple:
        - nodes (list of Node): List of nodes in the mesh.
        - elements (list of Element): List of elements in the mesh.
    """
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
    
    # Indicate Mesh creating is done
    print("\nMesh is created.")
    print(f"Total number of nodes is {len(nodes)}")
    print(f"Total number of elements is {len(elements)}")

    return nodes, elements

class System:
    """
    Represents the finite element system for wave propagation simulation.

    Attributes:
    domain_x (tuple of float): The x-axis boundaries of the domain (xmin, xmax).
    domain_y (tuple of float): The y-axis boundaries of the domain (ymin, ymax).
    num_elements_x (int): Number of elements along the x-axis.
    num_elements_y (int): Number of elements along the y-axis.
    nodes (list of Node): List of nodes in the mesh.
    elements (list of Element): List of elements in the mesh.
    num_nodes (int): Total number of nodes in the system.
    global_stiffness (numpy.ndarray): Global stiffness matrix.
    global_mass (numpy.ndarray): Global mass matrix.
    u_solved (numpy.ndarray or None): Solution matrix containing displacement values.
    left_boundary (list of Node): Nodes located on the left boundary.
    right_boundary (list of Node): Nodes located on the right boundary.
    top_boundary (list of Node): Nodes located on the top boundary.
    bottom_boundary (list of Node): Nodes located on the bottom boundary.
    """   
    def __init__(self, domain_x=None, domain_y=None, num_elements_x=None, num_elements_y=None, wavespeed=None, filename=None):
        """
        Initializes the System class.

        Parameters:
        domain_x (tuple of float, optional): X-axis boundaries of the domain.
        domain_y (tuple of float, optional): Y-axis boundaries of the domain.
        num_elements_x (int, optional): Number of elements along the x-axis.
        num_elements_y (int, optional): Number of elements along the y-axis.
        wavespeed (float, optional): Wave speed associated with the elements.
        filename (str, optional): File name to load a saved system configuration.
        """
        # Initialize the class with the save file.
        if filename:  # If a filename is provided, load the system from file
            self.load(filename)
        
        #Initialize the the class default
        else:
            self.domain_x = domain_x
            self.domain_y = domain_y
            self.num_elements_x = num_elements_x
            self.num_elements_y = num_elements_y
            self.nodes, self.elements = create_mesh(domain_x, domain_y, num_elements_x, num_elements_y, wavespeed)
            self.num_nodes = len(self.nodes)
            self.global_stiffness = np.zeros((self.num_nodes, self.num_nodes))
            self.global_mass = np.zeros((self.num_nodes, self.num_nodes))
            self.u_solved = None

            self.left_boundary, self.right_boundary, self.top_boundary, self.bottom_boundary = self.identify_boundary_nodes()

            self.assemble_global_matrices()

    def identify_boundary_nodes(self):
        """
        Identifies the nodes located at the boundaries of the domain.

        Returns:
        tuple:
            - left_boundary (list of Node): Nodes on the left boundary.
            - right_boundary (list of Node): Nodes on the right boundary.
            - top_boundary (list of Node): Nodes on the top boundary.
            - bottom_boundary (list of Node): Nodes on the bottom boundary.
        """
        left_boundary = []
        right_boundary = []
        top_boundary = []
        bottom_boundary = []
        
        for node in self.nodes:
            if node.x == self.domain_x[0]:      
                left_boundary.append(node)
            elif node.x == self.domain_x[1]:    
                right_boundary.append(node)
            if node.y == self.domain_y[1]:      
                top_boundary.append(node)
            elif node.y == self.domain_y[0]:    
                bottom_boundary.append(node)
                
        return left_boundary, right_boundary, top_boundary, bottom_boundary
    
    def get_middle_node(self):
        """
        Finds the node closest to the center of the domain.

        Returns:
        middle_node (Node): The middle node of the mesh.
        """
        # Calculate the midpoint coordinates of the domain
        x_mid = (self.domain_x[0] + self.domain_x[1]) / 2
        y_mid = (self.domain_y[0] + self.domain_y[1]) / 2

        # Find the node closest to the midpoint
        middle_node = min(
            self.nodes, 
            key=lambda node: (node.x - x_mid) ** 2 + (node.y - y_mid) ** 2
        )

        return middle_node
    
    def get_info(self, detailed = False):
        """
        Prints information about the system matrices and optionally details about the middle node.

        Parameters:
        detailed (bool, optional): If True, prints additional details about the middle node.
        """
        mass_size = self.global_mass.shape
        stiffness_size = self.global_stiffness.shape
        force_size = self.force_time_series.shape

        print(f"\nMass matrix size: {mass_size}")
        print(f"Stiffness matrix size: {stiffness_size}")
        print(f"Force vector size: {force_size}")

        if hasattr(self, 'u_solved'):
            u_solved_size = self.u_solved.shape
            print(f"Displacement (u_solved) size: {u_solved_size}\n")
        
        if detailed:
            # Calculate the index of the middle node
            middle_node_index = len(self.nodes) // 2
            print(f"Middle node index: {middle_node_index}")
            print(f"Middle node coordinates: ({self.nodes[middle_node_index].x}, {self.nodes[middle_node_index].y})")
    
    def assemble_global_matrices(self):
        """
        Prints information about the system matrices and optionally details about the middle node.

        Parameters:
        detailed (bool, optional): If True, prints additional details about the middle node.
        """
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

        print("\nGlobal matrices assembled")

    def apply_boundary_conditions(self):
        """
        Applies boundary conditions by modifying the global stiffness matrix and force vector.
        """
        for i in range(self.num_nodes):
            if (self.nodes[i].x == self.domain_x[0] or self.nodes[i].x == self.domain_x[1] or 
                self.nodes[i].y == self.domain_y[0] or self.nodes[i].y == self.domain_y[1]):
                
                self.global_stiffness[i, :] = 0
                self.global_stiffness[i, i] = 1
                self.force_vector[i] = 0
                
    def gaussian_initial_condition(self):
        """
        Generates an initial displacement field in the form of a Gaussian wave.

        Returns:
        initial_displacement (numpy.ndarray): Initial displacement values at each node.
        """
        x0 = (self.domain_x[0] + self.domain_x[1]) / 2
        y0 = (self.domain_y[0] + self.domain_y[1]) / 2
        A = 4
        sigma = 0.1
        initial_displacement = np.zeros(self.num_nodes)

        # Create mesh grid (wrt. nodes)
        x = np.linspace(self.domain_x[0], self.domain_x[1], self.num_elements_x + 1)
        y = np.linspace(self.domain_y[0], self.domain_y[1], self.num_elements_y + 1)
        xx, yy = np.meshgrid(x, y)

        # Gaussian displacements
        for i, node in enumerate(self.nodes):
            initial_displacement[i] = A * np.exp(-((node.x - x0)**2 + (node.y - y0)**2) / (2 * sigma**2))

        return initial_displacement


    def solve_with_initial_conditions(self, total_time, time_steps):
        """
        Solves the wave propagation system using the given initial conditions.

        Parameters:
        total_time (float): Total simulation time.
        time_steps (int): Number of time steps.

        Returns:
        u (numpy.ndarray): Solution matrix containing displacement values at each time step.
        """
        print("\nSOLVING IS STARTED")
        initial_displacement = self.gaussian_initial_condition()
        initial_velocity = np.zeros(self.num_nodes)
        u = np.zeros((self.num_nodes, time_steps))  # Displacement matrix( columns -> timesteps )
        v = initial_velocity  # Hız
        dt = total_time / time_steps
        mass_inv = np.linalg.inv(self.global_mass)  # Inverse of Mass matrix

        # First two timesteps
        u[:, 0] = initial_displacement
        u[:, 1] = u[:, 0] + dt * v

        for n in range(1, time_steps - 1):
            if n % (time_steps // 10) == 0:  
                progress = (n / (time_steps - 1)) * 100 
                print(f"{progress:.0f}% completed")
            u[:, n + 1] = (
                2 * u[:, n]
                - u[:, n - 1]
                + dt**2 * mass_inv @ (-self.global_stiffness @ u[:, n])
            )

        self.u_solved = u
        print("SOLVING IS DONE")
        return u
    
    # Function for saving the system variables after solving
    def save(self, filename):
        """
        Saves the system state to a file.

        Parameters:
        filename (str): Name of the file to save the system state.
        """
        data = {
            'nodes': self.nodes,
            'elements': self.elements,
            'num_elements_x': self.num_elements_x,
            'num_elements_y': self.num_elements_y,
            'global_stiffness': self.global_stiffness,
            'global_mass': self.global_mass,
            'left_boundary': self.left_boundary,
            'right_boundary': self.right_boundary,
            'top_boundary': self.top_boundary,
            'bottom_boundary': self.bottom_boundary,
            'domain_x': self.domain_x,
            'domain_y': self.domain_y,
            'num_nodes': self.num_nodes,
            'force_time_series': self.force_time_series,
            'u_solved': self.u_solved
        }

        # Save using numpy for matrices and pickling for the rest
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

        print(f"\nSystem saved successfully to {filename}.")

    # Loading the saved data file
    def load(self, filename):
        """
        Loads a previously saved system state from a file.

        Parameters:
        filename (str): Name of the file containing the saved system state.
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        self.nodes = data['nodes']
        self.elements = data['elements']
        self.num_elements_x = data['num_elements_x']
        self.num_elements_y = data['num_elements_y']
        self.global_stiffness = data['global_stiffness']
        self.global_mass = data['global_mass']
        self.left_boundary = data['left_boundary']
        self.right_boundary = data['right_boundary']
        self.top_boundary = data['top_boundary']
        self.bottom_boundary = data['bottom_boundary']
        self.domain_x = data['domain_x']
        self.domain_y = data['domain_y']
        self.num_nodes = data['num_nodes']
        self.force_time_series = data['force_time_series']
        self.u_solved = data['u_solved']

        print(f"\nSystem loaded successfully from {filename}.")
    

    def animate_wave_solution(self, time_steps):
        """
        Creates an animation of the wave propagation solution.

        Parameters:
        time_steps (int): Number of time steps used in the solution.
        """
        if self.u_solved is None:
            print('Error: Solve the system first.')
            return

        u = self.u_solved
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_coords = [node.x for node in self.nodes]
        y_coords = [node.y for node in self.nodes]
        X, Y = np.meshgrid(np.unique(x_coords), np.unique(y_coords))
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Colorbar axes

        # X & Y boundaries
        xMin, xMax = np.min(x_coords), np.max(x_coords)
        yMin, yMax = np.min(y_coords), np.max(y_coords)
        zMax = np.max(u[:, 0])

        # Surface for the first frame
        surface = ax.plot_surface(X, Y, np.zeros_like(X), cmap='viridis', edgecolor='k', linewidth=0.3)
        fig.colorbar(surface, cax=cbar_ax)  # Add colorbar

        def update_plot(frame):
            ax.clear()
            Z = np.zeros_like(X)
            for i, node in enumerate(self.nodes):
                Z.flat[i] = u[i, frame]

            surface = ax.plot_surface(
                X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.3)
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
            ax.set_zlim(-zMax, zMax)  # Z axis limit
            ax.set_title(f'Wave Propagation - Frame {frame + 1}/{time_steps}', fontsize=14, pad=20)

            # Update Colorbar
            #cbar_ax.clear()
            #fig.colorbar(surface, cax=cbar_ax)

            return ax

        ani = animation.FuncAnimation(fig, update_plot, frames=time_steps, blit=False, interval=0.000001)
        self.anim = ani
        plt.show()