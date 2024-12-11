import numpy as np
import math

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
    Generates a sinusoidal burst modulated by a windowed sinusoidal envelope.

    Parameters:
    frequency (float): The frequency of the sinusoidal burst (Hz).
    cycles (int): The number of cycles in the burst.
    amplitude (float): The amplitude of the sinusoidal burst.
    t (numpy.ndarray or float): The time(s) at which the burst is evaluated.

    Returns:
    numpy.ndarray or float: The value(s) of the sinusoidal burst at the given time(s).
    """
    
    omega = frequency * 2 * np.pi
    return (amplitude * 
            ((t <= cycles / frequency) & (t > 0)) * 
            np.sin(omega * t) * 
            (np.sin(omega * t / cycles)) ** 2)

    

