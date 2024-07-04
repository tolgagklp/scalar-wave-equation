import math
import numpy as np

def Gaussian(x, x0, sigma):
    return math.exp( - np.power(x - x0, 2) / (2 * np.power(sigma, 2)))

def analyticGaussianSolution(x, x0, t, c, sigma):
    u_left = Gaussian(x, x0 - c*t, sigma)
    u_right = Gaussian(x, x0 + c*t, sigma)
    return 0.5 *(u_left + u_right)

