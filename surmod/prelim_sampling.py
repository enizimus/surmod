import numpy as np
import matplotlib.pyplot as plt


def rand_orientation_mat(
    k: int, p: int, xi: int, seed: None
) -> (np.ndarray, np.ndarray):
    """Generates a random orientation matrix B*

    Arguments:
        k {int} -- grid dimension (how many dimensions grid has)
        p {int} -- grid level (how many points in grid)
        xi {int} -- elementary effect step size
    Returns:
        Bs {np.ndarray} -- Random orientation matrix B*
        Ps {np.ndarray} -- Matrix indicating which vars change
    """

    if seed != None:
        np.random.seed(seed)

    m = k + 1
    delta = xi / (p - 1)

    v = np.random.choice([-1, 1], size=k, p=[0.5, 0.5])
    Ds = np.diag(v)
    J = np.ones((m, k))
    B = np.tril(J, k=-1)

    sv = np.arange(0, (1 - delta) + 1 / (p - 1), 1 / (p - 1))
    xs = np.random.choice(sv, size=k, p=np.ones((len(sv),)) / len(sv))[None, :]

    ind = np.arange(k)
    np.random.shuffle(ind)
    Ps = np.eye(k)[:, ind]

    Bs = (J[:, 0][:, None] * xs + (delta / 2) * ((2 * B - J) @ Ds + J)) @ Ps

    return Bs, Ps


def sampling_matrix(
    k: int, p: int, xi: int, r: int, seed=None
) -> (np.ndarray, np.ndarray):
    """Generates a sampling matrix X consisting of r random
       orientation matrices B*

    Arguments:
        k {int} -- grid dimension (how many dimensions grid has)
        p {int} -- grid level (how many points in grid)
        xi {int} -- elementary effect step size
        r {int} -- number of elementary effects
    Returns:
        X {np.ndarray} -- r Random orientation matrices B* row concatenated
        P {np.ndarray} -- Matrix indicating which vars change
    """

    X = np.zeros(((k + 1) * r, k))
    P = np.zeros((k * r, k))

    for i in range(r):
        (
            X[i * (k + 1) : (i + 1) * (k + 1), :],
            P[i * k : (i + 1) * k, :],
        ) = rand_orientation_mat(k, p, xi, seed)

    return X, P


def sample_objective(X, f, ranges):

    # Scale to fit variable ranges :
    dranges = np.diff(ranges, axis=0)
    X = X * dranges + ranges[0, :]

    objectives = f(X)

    return objectives


def calc_elem_effects(
    objectives: np.ndarray, P: np.ndarray, k: int, r: int
) -> (np.ndarray, np.ndarray):
    """Generates a sampling plan and scales it with the ranges
    of the individual variables, then computes the elementary
    effects Matrix F and the mean and standard deviation of these
    values for each design variable.

    Arguments:
        k {int} -- grid dimension (how many dimensions grid has)
        p {int} -- grid level (how many points in grid)
        xi {int} -- elementary effect step size
        r {int} -- number of elementary effects
        f {function} -- objective function to evaluate
        ranges {2xk np.ndarray} -- contains ranges of individual variables
    Returns:
        F_mean {np.ndarray} -- mean values of elem. effect matrix F
        F_std {np.ndarray} -- std values of elem. effect matrix F
    """

    # Evaluate function at X and compute d_i(x)
    d = np.diff(objectives)
    T = objectives.reshape((r, k + 1)).T
    D = np.diff(T, axis=0)

    j = -1

    # Compute matrix F :
    F = np.zeros((k, r))

    for i in range(P.shape[0]):

        if i % k == 0:
            j += 1

        F[np.nonzero(P[i]), j] = D[i % k, j]

    # Compute mean and standard deviation
    F_mean = np.mean(F, axis=1)
    F_std = np.std(F, axis=1)

    return F_mean, F_std
