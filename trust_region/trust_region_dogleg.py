import numpy as np
import numpy.linalg as ln
import scipy as sp
from math import sqrt


# Objective function
def f(x):
    return 11 * x[0] ** 2 + 10 * x[1] ** 2 - 20 * x[0] * x[1] - 2 * x[0] + 1


# Gradient
def jac(x):
    return np.array([22 * x[0] - 20 * x[1] - 2, 20 * x[1] - 20 * x[0]])


# Hessian
def hess(x):
    return np.array([[22, -20], [-20, 20]])


def dogleg_method(Hk, gk, Bk, trust_radius):
    pB = -np.dot(Hk, gk)
    norm_pB = sqrt(np.dot(pB, pB))  # 书上式6.2

    if norm_pB <= trust_radius:
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = sqrt(dot_pU)

    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    pB_pU = pB - pU  # 书6.3式上一个公式
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU

    return pU + tau * pB_pU


def trust_region_dogleg(func, jac, hess, x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, gtol=1e-6,
                        maxiter=100):
    xk = x0
    trust_radius = initial_trust_radius
    k = 0
    while True:

        gk = jac(xk)
        Bk = hess(xk)
        Hk = np.linalg.inv(Bk)

        pk = dogleg_method(Hk, gk, Bk, trust_radius)

        act_red = func(xk) - func(xk + pk)

        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

        rhok = act_red / pred_red  # 书6.3式，接下来将利用书6.4式对rhok与n1，n2进行比较（0.25，0.75）
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red

        norm_pk = sqrt(np.dot(pk, pk))

        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else:
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk

        if ln.norm(gk) < gtol:
            break

        if k >= maxiter:
            break
        k = k + 1
        print(f"第{k}次迭代\n当前迭代点:{xk}")
    return xk


result = trust_region_dogleg(f, jac, hess, [0, 0])
print("Result of trust region dogleg method: {}".format(result))
print("Value of function at a point: {}".format(f(result)))
