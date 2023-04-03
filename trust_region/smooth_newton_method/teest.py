import math

import numpy as np


def phi(mu, a, b):
    p = a + b - math.sqrt(math.pow((a - b), 2) + 4 * math.pow(mu, 2))
    # p 为一个数
    return p


def dah(n, mu, lam, d, gk, Bk, deltak):
    H = np.zeros(((n + 2), 1))
    H[0] = mu
    H[1] = phi(mu, lam, deltak ** 2 - np.linalg.norm(d, ord=2) ** 2)
    I = np.identity(n)
    H[2:n+2] = np.dot((Bk + lam * I), d) + gk
    # H 为(n+2)*1
    return H


def JacobiH(n, mu, lam, d, Bk, deltak):
    J = np.zeros((n + 2, n + 2), dtype=float)
    I = np.identity(n)
    t2 = math.sqrt((lam + np.linalg.norm(d, ord=2) ** 2 - deltak ** 2) ** 2 + 4 * mu ** 2)
    pmu = -4 * mu / t2
    thetak = (lam + np.linalg.norm(d, ord=2) ** 2 - deltak ** 2) / t2
    # J = np.array([[1, 0, np.zeros(n, 1)],
    #               [pmu, 1 - thetak, -2 * (1 + thetak) * d.T],
    #               [np.zeros(n, 1), d, Bk + lam * I]])
    J[0][0] = 1
    J[0][1] = 0
    J[0][2:n+2] = np.zeros((1, n), dtype=float)
    J[1][0] = pmu
    J[1][1] = 1 - thetak
    J[1][2:n + 2] = -2 * (1 + thetak) * d.T
    J[2:n+2, 0:0] = np.zeros((n, 1), dtype=float)
    J[2:n+2, 1:1] = d
    J[2:n+2, 2:n+2] = Bk + lam * I
    # J 为n+2*n+2的矩阵
    return J


def psi(n, mu, lam, d, gk, Bk, deltak, gamma):
    H = dah(n, mu, lam, d, gk, Bk, deltak)
    si = gamma * np.linalg.norm(H, ord=2) * min(1.0, np.linalg.norm(H, ord=2))
    # si 为一个数
    return si


def smooth_newton_method(n,f, gk, Bk, deltak):
    beta = 0.6
    sigma = 0.2
    mu0 = 0.05
    lam0 = 0.05
    gamma = 0.05
    d0 = np.ones((n, 1))
    z0 = np.array([[mu0, lam0, d0.T]], dtype=object).T
    zbar = np.zeros(((n + 2), 1))
    zbar[0] = mu0
    i = 0
    z = z0
    mu = mu0
    lam = lam0
    d = d0
    while i <= 150:
        H = dah(n, mu, lam, d, gk, Bk, deltak)
        if np.linalg.norm(H) < 1e-8:
            break
        J = JacobiH(n, mu, lam, d, Bk, deltak)
        b = psi(n, mu, lam, d, gk, Bk, deltak, gamma) * zbar - H
        J_li = np.linalg.inv(J)
        dz = np.dot(J_li, b)
        dmu = dz[1]
        dlam = dz[2]
        dd = dz[2:n + 2]
        m = 0
        mi = 0
        while m < 20:
            t1 = beta ** m
            Hnew = dah(n, mu + t1 * dmu, lam + t1 * dlam, d + t1 * dd, gk, Bk, deltak)
            if np.linalg.norm(Hnew, ord=2) <= (1 - sigma * (1 - gamma * mu0) * pow(beta, m) * np.linalg.norm(H, ord=2)):
                mi = m
                break
            m = m + 1
        alpha = pow(beta, mi)
        mu = mu + alpha * dmu
        lam = lam + alpha * dlam
        d = d + alpha * dd
        i = i + 1
    return d


