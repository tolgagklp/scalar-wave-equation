import math
import numpy as np

def Gaussian(x, x0, sigma):
    return math.exp( - np.power(x - x0, 2) / (2 * np.power(sigma, 2)))

def analyticGaussianSolution(x, x0, t, c, sigma):
    u_left = Gaussian(x, x0 - c*t, sigma)
    u_right = Gaussian(x, x0 + c*t, sigma)
    return 0.5 *(u_left + u_right)

def analyticSolution2D(x, y, t):

    temp = 0
    for k in range(1, 41):
        for l in range(1, 41):
            term1 = (((-1) ** k) - 1) * ( ((-1) ** l) - 1)
            term2 = np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
            term3 = np.cos( np.sqrt( (k ** 2) + (l ** 2)) * np.pi * t)
            term4 = ((k ** 3) * (l ** 3))

            temp += (term1 * term2 * term3) / term4

    return (16 / (np.pi ** 2)) * temp
