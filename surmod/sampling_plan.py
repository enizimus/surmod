import numpy as np
from scipy.spatial.distance import pdist
import line_profiler

##@profile
def rlh(n: int, k: int, edges: int = 0):
    """Generate random latin hypercube with dimensions n x k

    Args:
        n (int): number of rows (samples)
        k (int): number of columns (variables)
        edges (int, optional): Affects the scaling of the entries. Defaults to 0.

    Returns:
        X (n x k np.ndarray): Random latin hypercube
    """
    X = np.zeros((n, k))
    for i in range(k):
        X[:, i] = np.random.permutation(n) + 1

    if edges == 1:
        X = (X - 1) / (n - 1)
    else:
        X = (X - 0.5) / n

    return X

#@profile
def compute_dists(X: np.ndarray, p: int = 1) -> (np.ndarray, np.ndarray):
    """Computes distances between all pairs of points, keeps the unique
    ones and computes how many each of them repeat.

    Args:
        X (np.ndarray): Sampling plan
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        unique_dist (np.ndarray): Unique distances between points
        J (np.ndarray): Contains for every distance in unique_dist how
        many times it repeats in the nonunique distances.
    """

    dist = pdist(X, "minkowski", p=p)
    unique_dist = np.unique(dist)

    # n_dist = len(dist)
    # n_udist = len(unique_dist)

    # J = np.count_nonzero(dist.repeat(n_udist).reshape(n_dist, n_udist) == unique_dist.repeat(n_dist).reshape(n_udist, n_dist).T, axis=0)

    J = np.zeros_like(unique_dist)

    for i, d in enumerate(unique_dist):
       J[i] = np.count_nonzero(dist == d)

    return J, unique_dist

#@profile
def morris_mitchel_criterion(X1: np.ndarray, X2: np.ndarray, p: int = 1) -> (int):
    """Implements the Morris Mitchel criterion based on the distances between
    the points in the sampling plans.

    Args:
        X1 (np.ndarray): Sampling plan 1
        X1 (np.ndarray): Sampling plan 2
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        (int): 0 if sampling plans equal
               1 if X1 better
               2 if X2 better
    """
    cond = 0

    I1 = np.argsort(X1[:, 0])
    I2 = np.argsort(X2[:, 0])

    if not np.array_equal(X1[I1, :], X2[I2, :]):

        J1, d1 = compute_dists(X1, p)
        J2, d2 = compute_dists(X2, p)

        m = min(len(J1), len(J2))

        Jd1 = np.zeros((m,))
        Jd2 = np.zeros((m,))

        if m % 2 == 0:
            f1 = m // 2
            f2 = f1
        else:
            f1 = m // 2 + 1
            f2 = f1 - 1

        Jd1[0:m:2] = d1[0:f1]
        Jd1[1:m:2] = J1[0:f2]

        Jd2[0:m:2] = d2[0:f1]
        Jd2[1:m:2] = J2[0:f2]

        c = (Jd1 > Jd2) + 2 * (Jd1 < Jd2)

        if not np.sum(c) == 0:
            i = 0
            while c[i] == 0:
                i += 1

            cond = c[i]

    return cond

##@profile
def morris_mitchel_phi_criterion(X: np.ndarray, q: int = 2, p: int = 1):
    """Computes the Morris Mitchel Phi criterion where the space filling
    property of the latin hypercube is expressed through the Parameter
    Phi -> lower is better.

    Args:
        X (np.ndarray): Sampling plan
        q (int, optional): Phi criterion parameter. Defaults to 2.
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        [float]: phi value for input sampling plan.
    """
    J, d = compute_dists(X, p)
    return np.dot(J, d ** (-q)) ** (1 / q)

##@profile
def sort_morris_mitchel(X3D: np.ndarray, p: int = 1) -> (np.ndarray):
    """Sorts the 3-dimensional array of sampling plans according
    to the Morris Mitchel criterion.

    Args:
        X3D (np.ndarray): 3-dimensional array of 2D sampling plans
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        (np.ndarray): Indices of sorted 3D array.
    """
    nplans = X3D.shape[0]
    indices = np.arange(nplans)

    is_swapping = True

    while is_swapping:
        is_swapping = False
        i = 0
        while i < nplans - 1:
            if (
                morris_mitchel_criterion(
                    X3D[indices[i], :, :], X3D[indices[i + 1], :, :], p
                )
                == 2
            ):

                indices[[i, i + 1]] = indices[[i + 1, i]]
                is_swapping = True
            i += 1

    return indices

##@profile
def sort_morris_mitchel_phi(X3D: np.ndarray, p: int = 1, q: int = 2):
    """Sorts the 3-dimensional array of sampling plans according
    to the Morris Mitchel Phi criterion.

    Args:
        X3D (np.ndarray): 3-dimensional array of 2D sampling plans
        p (int, optional): Degree of norm. Defaults to 1.
        q (int, optional): Phi criterion parameter. Defaults to 2.

    Returns:
        (np.ndarray): Indices of sorted 3D array.
    """
    Phi = np.array([morris_mitchel_phi_criterion(X, p, q) for X in X3D])
    indices = np.argsort(Phi)

    return indices


def perturb_plan(X: np.ndarray, num_perturb: int = 1) -> (np.ndarray):
    """Perturbs the input sampling plan by performing smallest possible
    alteration to latin hypercube without leaving the latin hypercube
    space.

    Args:
        X (np.ndarray): Sampling plan
        num_perturb (int, optional): Number of perturbations to perform. Defaults to 1.

    Returns:
        np.ndarray: Perturbed sampling plan
    """
    n = np.arange(X.shape[0])
    k = np.arange(X.shape[1])

    for i in range(num_perturb):

        i_col = np.random.choice(k)
        i_row1, i_row2 = 0, 0

        while i_row1 == i_row2:
            i_row1 = np.random.choice(n)
            i_row2 = np.random.choice(n)

        X[[i_row1, i_row2], i_col] = X[[i_row2, i_row1], i_col]

    return X

##@profile
def evolve_lh(X: np.ndarray, n_children: int, n_iter: int, q: int = 2) -> (np.ndarray):
    """Evolutionary process optimization of input sampling plan.

    Args:
        X (np.ndarray): Sampling plan.
        n_children (int): Number of offspring each iteration.
        n_iter (int): Number of iterations to perform.
        q (int, optional): Phi criterion parameter. Defaults to 2.

    Returns:
        (np.ndarray): Optimized latin hypercube
    """
    n = X.shape[0]

    X_best = X
    phi_best = morris_mitchel_phi_criterion(X)

    lvl_off = np.floor(0.85 * n_iter)

    for i in range(1, n_iter + 1):

        if i < lvl_off:
            mutations = int(np.round(1 + (0.5 * n - 1) * (lvl_off - i) / (lvl_off - 1)))
        else:
            mutations = 1

        X_improved = X_best
        phi_improved = phi_best

        for _ in range(n_children):

            X_child = perturb_plan(X_best.copy(), mutations)
            phi_child = morris_mitchel_phi_criterion(X_child, q)

            if phi_child < phi_improved:
                X_improved = X_child
                phi_improved = phi_child

        if phi_improved < phi_best:
            X_best = X_improved
            phi_best = phi_improved

    return X_best

import concurrent.futures

#@profile
def get_evolved_lh(
    n: int, k: int, n_children: int = 10, n_iter: int = 10, p: int = 1
) -> (np.ndarray):
    """Starts with random hypercubes for every q value in [1, 2, 5, 10, 20, 50, 100]
    and uses evolve_lh to compute optimal sampling plan for ever q. Then sort these
    by the Morris Mitchel criterion and return the best (first).

    Args:
        n (int): number of rows (samples)
        k (int): number of columns (variables)
        n_children (int, optional): Number of offspring each iteration. Defaults to 10.
        n_iter (int, optional): Number of iterations to perform. Defaults to 10.
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        np.ndarray: Returns optimal sampling plan.
    """
    q_arr = np.array([1, 2, 5, 10, 20, 50, 100])  # as proposed in paper

    X3D = np.zeros((len(q_arr), n, k))
    X_start = rlh(n, k)

    # for i in range(len(q_arr)):
    #     X3D[i, :, :] = evolve_lh(X_start, n_children, n_iter, q_arr[i])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(evolve_lh, X_start, n_children, n_iter, q) for q in q_arr]

        for ind, r in enumerate(results):
            X3D[ind, :, :] = r.result()

    indices = sort_morris_mitchel(X3D, p)

    X_best = X3D[indices[0]]

    print("Best LH found for q = {}.".format(q_arr[indices[0]]))

    return X_best

import datetime

if __name__ == '__main__':

    print(datetime.datetime.now())
    X = get_evolved_lh(2000, 6, 5, 5)

    print(datetime.datetime.now())