import numpy as np
import math
import scipy

def Gaussian(x, x0, sigma):
  return math.exp(-np.power(x-x0, 2) / 2 / np.power(sigma, 2) )

def GaussianOverlay(x, x0, sigma, c, t):
  u_right = Gaussian(x, x0 + c*t, sigma)
  u_left = Gaussian(x, x0 - c*t, sigma)

  return u_right + u_left

def generate_SinBurst(frequency, cycles, amplitude,t):
    omega = frequency * 2 * np.pi

    return amplitude * ((t<= cycles / frequency) & (t>0)) * np.sin(omega * t) * (np.sin(omega * t / 2/cycles )) ** 2

def generateSineBurst(frequency, cycles, amplitude, t, Nx, N):
    omega = frequency * 2 * np.pi
    f = np.zeros((Nx - 2, N + 1))
    for t_val in range(len(t)):
        if t[t_val] < t[-1] // 4:
            f[49, t_val] = (((amplitude / 2 * np.sin(omega * t[t_val])) + (
                        amplitude / 2 * np.sin(omega * 0.7 * t[t_val]))) * np.sin(
                (2 * np.pi / (t[-1] // 4) / 4) * t[t_val]) ** 2)  # normalization over the applied area
        else:
            f[49, t_val] = ((amplitude / 2 * np.sin(omega * t[t_val])) + (
                        amplitude / 2 * np.sin(omega * 0.7 * t[t_val])))  # normalization over the applied area
    return f


def newmark(M, C, K, F, u0, ut0, nt, dt, gaama=1 / 2, beta=1 / 4):
    n = M.shape[0]

    a0 = 1 / (beta * (dt ** 2))
    a1 = gaama / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = (1 / (2 * beta)) - 1
    a4 = (gaama / beta) - 1
    a5 = (dt / 2) * ((gaama / beta) - 2)
    a6 = dt * (1 - gaama)
    a7 = gaama * dt

    depl = np.zeros((n, nt + 1))
    vel = np.zeros((n, nt + 1))
    accl = np.zeros((n, nt + 1))

    depl[:, 0] = u0
    vel[:, 0] = ut0
    accl[:, 0] = np.linalg.inv(M) @ (F[:, 0] - C @ vel[:, 0] - K @ depl[:, 0])

    Kcap = K + a0 * M + a1 * C
    lu, piv = scipy.linalg.lu_factor(Kcap)

    a = a1 * C + a0 * M
    b = a4 * C + a2 * M
    c = a5 * C + a3 * M

    for i in range(1, nt):
        Fcap = F[:, i] + a @ depl[:, i - 1] + b @ vel[:, i - 1] + c @ accl[:, i - 1]
        depl[:, i] = scipy.linalg.lu_solve((lu, piv), Fcap)
        accl[:, i] = a0 * (depl[:, i] - depl[:, i - 1]) - a2 * vel[:, i - 1] - a3 * accl[:, i - 1]
        vel[:, i] = vel[:, i - 1] + a6 * accl[:, i - 1] + a7 * accl[:, i]

    return depl, vel, accl

