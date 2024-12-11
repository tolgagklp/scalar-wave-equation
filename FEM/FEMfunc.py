import numpy as np
import math
import scipy

def Gaussian(x, x0, sigma):
    """
    Computes the value of a Gaussian function.

    Parameters:
    x (float): The evaluation point.
    x0 (float): The center of the Gaussian function.
    sigma (float): The standard deviation of the Gaussian function.

    Returns:
    float: The value of the Gaussian function at x.
    """

    return math.exp(-np.power(x - x0, 2) / (2 * np.power(sigma, 2)))

def GaussianOverlay(x, x0, sigma, c, t):
    """
    Computes the sum of two Gaussian functions propagating in opposite directions.

    Parameters:
    x (float): The evaluation point.
    x0 (float): The center of the Gaussian functions at time t=0.
    sigma (float): The standard deviation of the Gaussian functions.
    c (float): The wave propagation speed.
    t (float): The time for which the overlay is evaluated.

    Returns:
    float: The combined value of the right- and left-moving Gaussian functions at x.
    """

    u_right = Gaussian(x, x0 + c * t, sigma)
    u_left = Gaussian(x, x0 - c * t, sigma)

    return u_right + u_left

def generate_SinBurst(frequency, cycles, amplitude, t):
    """
    Generates a sinusoidal burst modulated by a sinusoidal envelope.

    Parameters:
    frequency (float): The frequency of the sinusoidal burst (Hz).
    cycles (int): The number of cycles in the burst.
    amplitude (float): The amplitude of the sinusoidal burst.
    t (numpy.ndarray or float): The time(s) at which the burst is evaluated.

    Returns:
    numpy.ndarray or float: The value(s) of the sinusoidal burst at the given time(s).
    """

    omega = frequency * 2 * np.pi
    return amplitude * ((t <= cycles / frequency) & (t > 0)) * np.sin(omega * t) * (np.sin(omega * t / 2 / cycles)) ** 2

def generateSineBurst(frequency, cycles, amplitude, t, Nx, N):
    """
    Generates a spatially distributed sinusoidal burst over a domain.

    Parameters:
    frequency (float): The frequency of the sinusoidal burst (Hz).
    cycles (int): The number of cycles in the burst.
    amplitude (float): The amplitude of the sinusoidal burst.
    t (numpy.ndarray): Array of time steps for the burst.
    Nx (int): Number of spatial nodes.
    N (int): Number of time steps.

    Returns:
    numpy.ndarray: A 2D array containing the burst values over space and time.
    """

    omega = frequency * 2 * np.pi
    f = np.zeros((Nx - 2, N + 1))
    for t_val in range(len(t)):
        if t[t_val] < t[-1] // 4:
            f[49, t_val] = (((amplitude / 2 * np.sin(omega * t[t_val])) + (
                        amplitude / 2 * np.sin(omega * 0.7 * t[t_val]))) * 
                        np.sin((2 * np.pi / (t[-1] // 4) / 4) * t[t_val]) ** 2)  # normalization over the applied area
        else:
            f[49, t_val] = ((amplitude / 2 * np.sin(omega * t[t_val])) + (
                        amplitude / 2 * np.sin(omega * 0.7 * t[t_val])))  # normalization over the applied area
    return f

def newmark(M, C, K, F, u0, ut0, nt, dt, gaama=1 / 2, beta=1 / 4):
    """
    Implements the Newmark time integration method for solving second-order ODEs.

    Parameters:
    M (numpy.ndarray): Mass matrix.
    C (numpy.ndarray): Damping matrix.
    K (numpy.ndarray): Stiffness matrix.
    F (numpy.ndarray): External force matrix with dimensions (n, nt+1).
    u0 (numpy.ndarray): Initial displacement vector.
    ut0 (numpy.ndarray): Initial velocity vector.
    nt (int): Number of time steps.
    dt (float): Time step size.
    gaama (float): Newmark gamma parameter (default is 1/2, average acceleration method).
    beta (float): Newmark beta parameter (default is 1/4, average acceleration method).

    Returns:
    tuple: (depl, vel, accl)
        - depl (numpy.ndarray): Displacement values over time.
        - vel (numpy.ndarray): Velocity values over time.
        - accl (numpy.ndarray): Acceleration values over time.
    """
    
    n = M.shape[0]

    # Precompute constants
    a0 = 1 / (beta * (dt ** 2))
    a1 = gaama / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = (1 / (2 * beta)) - 1
    a4 = (gaama / beta) - 1
    a5 = (dt / 2) * ((gaama / beta) - 2)
    a6 = dt * (1 - gaama)
    a7 = gaama * dt

    # Initialize arrays for displacement, velocity, and acceleration
    depl = np.zeros((n, nt + 1))
    vel = np.zeros((n, nt + 1))
    accl = np.zeros((n, nt + 1))

    # Initial conditions
    depl[:, 0] = u0
    vel[:, 0] = ut0
    accl[:, 0] = np.linalg.inv(M) @ (F[:, 0] - C @ vel[:, 0] - K @ depl[:, 0])

    # Effective stiffness matrix
    Kcap = K + a0 * M + a1 * C
    lu, piv = scipy.linalg.lu_factor(Kcap)

    # Precompute terms
    a = a1 * C + a0 * M
    b = a4 * C + a2 * M
    c = a5 * C + a3 * M

    # Time-stepping loop
    for i in range(1, nt):
        Fcap = F[:, i] + a @ depl[:, i - 1] + b @ vel[:, i - 1] + c @ accl[:, i - 1]
        depl[:, i] = scipy.linalg.lu_solve((lu, piv), Fcap)
        accl[:, i] = a0 * (depl[:, i] - depl[:, i - 1]) - a2 * vel[:, i - 1] - a3 * accl[:, i - 1]
        vel[:, i] = vel[:, i - 1] + a6 * accl[:, i - 1] + a7 * accl[:, i]

    return depl, vel, accl
