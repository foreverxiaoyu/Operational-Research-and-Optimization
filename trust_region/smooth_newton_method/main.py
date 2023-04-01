import numpy as np
from teest import smooth_newton_method


def trust_region_subproblem(n, f, grad_f, hess_f, delta):

    return x


# Problem 1: min q(x) = 2x2_1 - 4x1x2 + 4x2^2 - 6x1 - 3x2
def f1(x):
    return 2 * x[0] ** 2 - 4 * x[0] * x[1] + 4 * x[1] ** 2 - 6 * x[0] - 3 * x[1]


def grad_f1(x):
    return np.array([4 * x[0] - 4 * x[1] - 6, 8 * x[1] - 4 * x[0] - 3])


def hess_f1(x):
    return np.array([[4, -4], [-4, 8]])


delta_values = [1, 2, 5]
n = 2
for delta in delta_values:
    x_opt = trust_region_subproblem(n, f1, grad_f1, hess_f1, delta)
    print("Problem 1, delta =", delta)
    print("Optimal solution:", x_opt)
    print("Optimal value:", f1(x_opt))

def f2(x):
    A = np.array([[3, -1, 2],
                  [-1, 2, 0],
                  [2, 0, 4]])
    b = np.array([[1],
                  [-3],
                  [-2]])
    x = np.array([x[0],
                   x[1],
                   x[2]])
    return 0.5 * np.dot(np.dot(x.T, A), x) + np.dot(b.T, x)


def grad_f2(x):
    A = np.array([[3, -1, 2],
                  [-1, 2, 0],
                  [2, 0, 4]])
    b = np.array([[1],
                  [-3],
                  [-2]])
    x = np.array([x[0],
                   x[1],
                   x[2]])
    return np.dot(A, x) + b


def hess_f2(x):
    A = np.array([[3, -1, 2],
                  [-1, 2, 0],
                  [2, 0, 4]])
    return A


delta_values = [1, 2, 5]
n = 3
for delta in delta_values:
    x_opt = trust_region_subproblem(n, f2, grad_f2, hess_f2, delta)
    print("Problem 2, delta =", delta)
    print("Optimal solution:", x_opt)
    print("Optimal value:", f2(x_opt))
