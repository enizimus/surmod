import numpy as np


def expfun(N):
    x1 = np.linspace(-2, 2, N)
    x2 = np.linspace(-2, 2, N)
    xx, yy = np.meshgrid(x1, x2)

    f = np.exp(1 - xx ** 2 - yy ** 2)

    return xx, yy, f


def cossin(N):
    x1 = np.linspace(-2, 2, N)
    x2 = np.linspace(-2, 2, N)
    xx, yy = np.meshgrid(x1, x2)

    f = np.cos(xx) * np.sin(yy)

    return xx, yy, f


def rosenbrock(N):
    x1 = np.linspace(-2, 2, N)
    x2 = np.linspace(-1, 3, N)
    xx, yy = np.meshgrid(x1, x2)
    f = 100 * (yy - xx ** 2) ** 2 + (1 - xx) ** 2

    return xx, yy, f


def rastrigin(N):
    x1 = np.linspace(-3, 3, N)
    x2 = np.linspace(-3, 3, N)
    xx, yy = np.meshgrid(x1, x2)
    f = (
        20
        + xx ** 2
        - 10 * np.cos(2 * np.pi * xx)
        + yy ** 2
        - 10 * np.cos(2 * np.pi * yy)
    )

    return xx, yy, f


def goldstein_price(N):
    x1 = np.linspace(-1.5, 1.5, N)
    x2 = np.linspace(-1.5, 1.5, N)
    xx, yy = np.meshgrid(x1, x2)
    f = (
        1
        + (xx + yy + 1) ** 2
        * (19 - 14 * xx + 3 * xx ** 2 - 14 * yy + 6 * xx * yy + 3 * yy ** 2)
    ) * (
        30
        + (2 * xx - 3 * yy) ** 2
        * (18 - 32 * xx + 12 * xx ** 2 + 48 * yy - 36 * xx * yy + 27 * yy ** 2)
    )

    return xx, yy, f


def booth(N):
    x1 = np.linspace(-3, 3, N)
    x2 = np.linspace(-3, 3, N)
    xx, yy = np.meshgrid(x1, x2)
    f = (xx + 2 * yy - 7) ** 2 + (2 * xx + yy - 5) ** 2

    return xx, yy, f
