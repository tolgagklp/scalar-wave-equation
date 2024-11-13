import numpy as np
import math

def Cosine(x, x0, omega):
  return math.cos(omega * (x-x0))
  
def CosineOverlay(x, x0, omega, c, t):
  u_right = Cosine(x, x0 + c*t, omega)
  u_left = Cosine(x, x0 - c*t, omega)
  return 0.5*(u_right + u_left)

def Gaussian(x, x0, sigma):
  return math.exp(-np.power(x-x0, 2) / 2 / np.power(sigma, 2) )

def GaussianOverlay(x, x0, sigma, c, t):
  u_right = Gaussian(x, x0 + c*t, sigma)
  u_left = Gaussian(x, x0 - c*t, sigma)
  return u_right + u_left

  
def partitionGP(GPp, GPw, node1, node2):
    GPp_return = [node1 + (p + 1) * (node2 - node1) / 2 for p in GPp]
    # weight is adjusted for smaller support of the inetgration
    GPw_return = [w * (node2 - node1) / 2.0 for w in GPw]
    return GPp_return, GPw_return


def generate_SinBurst(frequency, cycles, amplitude,t):
    omega = frequency * 2 * np.pi

    return amplitude * ((t<= cycles / frequency) & (t>0)) * np.sin(omega * t) * (np.sin(omega * t / cycles )) ** 2

    

