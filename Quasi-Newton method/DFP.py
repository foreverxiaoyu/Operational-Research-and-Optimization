import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt


def f(x):
    return x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 4 * x[0]


def f1(x):
    return np.array([2 * x[0] - 2 * x[1] - 4, 4 * x[1] - 2 * x[0]])


def DFP_method(f, gk, x0, maxiter=None, epsi=10e-3):
    if maxiter is None:
        maxiter = len(x0) * 200
    # 初始化
    count = 0
    gfk = gk(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)  # 初始化HK为单位阵
    HK = I
    xk = x0

    while ln.norm(gfk) > epsi and count < maxiter:
        dk = -np.dot(HK, gfk)  # dk = -gk*bk**-1
        line_search = sp.optimize.line_search(f, f1, xk, dk)  # 用wolfe准则求步长
        alpha_k = line_search[0]

        xkp1 = xk + alpha_k * dk  # 进行迭代
        sk = xkp1 - xk  # sk是位移差
        xk = xkp1

        gfkp1 = gk(xkp1)
        yk = gfkp1 - gfk  # yk是梯度差
        gfk = gfkp1

        count += 1

        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        HK = np.dot(A1, np.dot(HK, A2)) + (ro * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])
        # ---------------------------------------------------------------------------------------
        # 可视化
        x_store = np.zeros((1, 2))  # storing x values
        x_store[0, :] = xkp1
        x_store = np.append(x_store, [xkp1], axis=0)  # storing x
        x1 = np.linspace(min(x_store[:, 0] - 0.5), max(x_store[:, 0] + 0.5), 30)
        x2 = np.linspace(min(x_store[:, 1] - 0.5), max(x_store[:, 1] + 0.5), 30)
        X1, X2 = np.meshgrid(x1, x2)
        Z = f([X1, X2])
        plt.figure()
        plt.title('OPTIMAL AT: ' + str(x_store[-1, :]) + '\n IN ' + str(count) + ' ITERATIONS')
        plt.contourf(X1, X2, Z, 30, cmap='jet')
        plt.colorbar()
        plt.plot(x_store[:, 0], x_store[:, 1], c='w')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()

    return xk, count


result, k = DFP_method(f, f1, np.array([1, 1]))
z = f(result)

print('x_Result : {}'.format(result))
print('Iteration Count: {}'.format(k))
print('min_f(x) : {}'.format(z))
