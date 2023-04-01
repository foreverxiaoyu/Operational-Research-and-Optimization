import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import matplotlib.animation as animation
import math

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
camera = Camera(fig)

gr = (math.sqrt(5) + 1) / 2


def gss(f, a, b, tol=0.2, yint=[-10, 10]):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    """
    x = np.arange(a, b, 0.1)
    plt.xlim([a - 10, b + 10])
    plt.ylim(yint)
    plt.plot(x, f(x), color="red")
    plt.vlines(a, ymin=yint[0], ymax=len(x), colors='green', ls='--', lw=2,
               label='vline_multiple - full height')
    plt.vlines(b, ymin=yint[0], ymax=len(x), colors='blue', ls='--', lw=2,
               label='vline_multiple - full height')
    camera.snap()
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c) < f(d):  # f(c) > f(d) to find the maximum
            b = d
            plt.plot(x, f(x), color="red")
            plt.vlines(a, ymin=yint[0], ymax=len(x), colors='green', ls='--', lw=2,
                       label='vline_multiple - full height')
            plt.vlines(b, ymin=yint[0], ymax=len(x), colors='blue', ls='--', lw=2,
                       label='vline_multiple - full height')
            camera.snap()
        else:
            a = c
            plt.plot(x, f(x), color="red")
            plt.vlines(a, ymin=yint[0], ymax=len(x), colors='green', ls='--', lw=2,
                       label='vline_multiple - full height')
            plt.vlines(b, ymin=yint[0], ymax=len(x), colors='blue', ls='--', lw=2,
                       label='vline_multiple - full height')
            camera.snap()

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    animation = camera.animate()
    animation.save("gss.gif", writer='pillow', fps=6)
    return (b + a) / 2


if __name__ == '__main__':
    min = gss(lambda x: 2*x**2 + 50*np.sin(x), -10, 10, tol=0.1, yint=[-100, 100])
    print(min)
    plt.show()
