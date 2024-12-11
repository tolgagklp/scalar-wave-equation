import math
import numpy as np

def Gaussian(x, x0, sigma):
    """
    Computes the value of a Gaussian function.

    Parameters:
        x (float): The position where the function is evaluated.
        x0 (float): The mean or center of the Gaussian function.
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        float: The value of the Gaussian function at x.
    """
    
    return math.exp(-np.power(x - x0, 2) / (2 * np.power(sigma, 2)))

def analyticGaussianSolution(x, x0, t, c, sigma):
    """
    Computes the analytical solution of a Gaussian function for a wave equation.

    Parameters:
        x (float): The spatial position where the function is evaluated.
        x0 (float): The initial position of the Gaussian center.
        t (float): Time at which the solution is evaluated.
        c (float): Wave speed.
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        float: The value of the analytical Gaussian solution at (x, t).
    """

    u_left = Gaussian(x, x0 - c * t, sigma)
    u_right = Gaussian(x, x0 + c * t, sigma)
    return 0.5 * (u_left + u_right)

def analyticSolution2D(x, y, t):
    """
    Computes the analytical solution for a specific 2D wave equation with a specific initial condition.

    Parameters:
        x (float): The x-coordinate of the evaluation point.
        y (float): The y-coordinate of the evaluation point.
        t (float): Time at which the solution is evaluated.

    Returns:
        float: The value of the analytical solution at (x, y, t).
    """

    temp = 0
    for k in range(1, 41):
        for l in range(1, 41):
            term1 = (((-1) ** k) - 1) * (((-1) ** l) - 1)
            term2 = np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
            term3 = np.cos(np.sqrt((k ** 2) + (l ** 2)) * np.pi * t)
            term4 = ((k ** 3) * (l ** 3))

            temp += (term1 * term2 * term3) / term4

    return (16 / (np.pi ** 6)) * temp
