import numpy as np


def fun(x):
    return 3 * x[0] ** 2 / 2 + 1 * x[1] ** 2 / 2 - x[0] * x[1] - 2 * x[0]


def grad_f(x):
    return np.array([3 * x[0] - x[1] - 2, x[1] - x[0]])


def armijo(f, gf, x_k, d_k, beta=0.5, sigma=0.1, xk=None):
    m = 0
    m_k = 0
    while True:
        if f(x_k + beta ** m * d_k) > f(x_k) + sigma * beta ** m * gf(x_k).T @ d_k:
            m += 1
        else:
            m_k = m
            break
    alpha = pow(beta, m_k)
    return alpha


def FR_CG(f, grad_f, x0, eps=1e-5, max_iter=5e5):
    x = x0
    g = grad_f(x)
    d = -g
    k = 0
    while np.linalg.norm(g) > eps and k < max_iter:
        alpha = armijo(f, grad_f, x, d)
        x = x + alpha * d
        g_new = grad_f(x)
        beta = np.linalg.norm(g_new)*1.0 / np.linalg.norm(g)
        d = -g_new + beta * d
        g = g_new
        k += 1
    return x


x0 = np.array([-2, 4])
x_min = FR_CG(fun, grad_f, x0)
print("Minimum point: ", x_min)
print("Minimum value: ", fun(x_min))
