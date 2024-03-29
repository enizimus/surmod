from typing import Tuple
import numpy as np
import torch
from torch._C import dtype
# from scipy.spatial.distance import pdist

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def rlh(n: int, k: int, edges: int = 0) -> torch.Tensor:
    """Generate random latin hypercube with dimensions n x k

    Args:
        n (int): number of rows (samples)
        k (int): number of columns (variables)
        edges (int, optional): Affects the scaling of the entries. Defaults to 0.

    Returns:
        X (n x k torch.Tensor): Random latin hypercube
    """
    X:torch.Tensor = torch.zeros(n, k).to(dev)

    for i in range(k):
        X[:, i] = torch.randperm(n) + 1

    if edges == 1:
        X = (X - 1) / (n - 1)
    else:
        X = (X - 0.5) / n

    return X


def compute_dists(X: torch.Tensor, p: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes distances between all pairs of points, keeps the unique
    ones and computes how many each of them repeat.

    Args:
        X (torch.Tensor): Sampling plan
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        unique_dist (torch.Tensor): Unique distances between points
        J (torch.Tensor): Contains for every distance in unique_dist how
        many times it repeats in the nonunique distances.
    """

    dist = torch.nn.functional.pdist(X, p=2)

    unique_dist = torch.unique(dist)
    J = torch.zeros_like(unique_dist)

    for i, d in enumerate(unique_dist):
        I = dist == d
        J[i] = torch.sum(I)

    return J, unique_dist


def morris_mitchel_criterion(X1: torch.Tensor, X2: torch.Tensor, p: int = 1) -> (int):
    """Implements the Morris Mitchel criterion based on the distances between
    the points in the sampling plans.

    Args:
        X1 (torch.Tensor): Sampling plan 1
        X1 (torch.Tensor): Sampling plan 2
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        (int): 0 if sampling plans equal
               1 if X1 better
               2 if X2 better
    """
    cond = 0

    I1 = torch.argsort(X1[:, 0])
    I2 = torch.argsort(X2[:, 0])

    if not torch.equal(X1[I1, :], X2[I2, :]):

        J1, d1 = compute_dists(X1, p)
        J2, d2 = compute_dists(X2, p)

        m = min(len(J1), len(J2))

        Jd1 = torch.zeros(m)
        Jd2 = torch.zeros(m)

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

        if not torch.sum(c) == 0:
            i = 0
            while c[i] == 0:
                i += 1

            cond = c[i]

    return cond


def morris_mitchel_phi_criterion(X: torch.Tensor, q: int = 2, p: int = 1):
    """Computes the Morris Mitchel Phi criterion where the space filling
    property of the latin hypercube is expressed through the Parameter
    Phi -> lower is better.

    Args:
        X (torch.Tensor): Sampling plan
        q (int, optional): Phi criterion parameter. Defaults to 2.
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        [float]: phi value for input sampling plan.
    """
    J, d = compute_dists(X, p)
    return (torch.dot(J, d ** (-q)) ** (1 / q)).item()


def sort_morris_mitchel(X3D: torch.Tensor, p: int = 1) -> (torch.Tensor):
    """Sorts the 3-dimensional array of sampling plans according
    to the Morris Mitchel criterion.

    Args:
        X3D (torch.Tensor): 3-dimensional array of 2D sampling plans
        p (int, optional): Degree of norm. Defaults to 1.

    Returns:
        (torch.Tensor): Indices of sorted 3D array.
    """
    nplans = X3D.shape[0]
    indices = torch.arange(nplans)

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


def sort_morris_mitchel_phi(X3D: torch.Tensor, p: int = 1, q: int = 2):
    """Sorts the 3-dimensional array of sampling plans according
    to the Morris Mitchel Phi criterion.

    Args:
        X3D (torch.Tensor): 3-dimensional array of 2D sampling plans
        p (int, optional): Degree of norm. Defaults to 1.
        q (int, optional): Phi criterion parameter. Defaults to 2.

    Returns:
        (torch.Tensor): Indices of sorted 3D array.
    """
    Phi = torch.tensor([morris_mitchel_phi_criterion(X, p, q) for X in X3D], dtype=torch.float32)
    indices = torch.argsort(Phi)

    return indices


def perturb_plan(X: torch.Tensor, num_perturb: int = 1) -> (torch.Tensor):
    """Perturbs the input sampling plan by performing smallest possible
    alteration to latin hypercube without leaving the latin hypercube
    space.

    Args:
        X (torch.Tensor): Sampling plan
        num_perturb (int, optional): Number of perturbations to perform. Defaults to 1.

    Returns:
        torch.Tensor: Perturbed sampling plan
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


def evolve_lh(X: torch.Tensor, n_children: int, n_iter: int, q: int = 2) -> (torch.Tensor):
    """Evolutionary process optimization of input sampling plan.

    Args:
        X (torch.Tensor): Sampling plan.
        n_children (int): Number of offspring each iteration.
        n_iter (int): Number of iterations to perform.
        q (int, optional): Phi criterion parameter. Defaults to 2.

    Returns:
        (torch.Tensor): Optimized latin hypercube
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

            X_child = perturb_plan(X_best.clone().detach(), mutations)
            phi_child = morris_mitchel_phi_criterion(X_child, q)

            if phi_child < phi_improved:
                X_improved = X_child
                phi_improved = phi_child

        if phi_improved < phi_best:
            X_best = X_improved
            phi_best = phi_improved

    return X_best

def get_evolved_lh(
    n: int, k: int, n_children: int = 10, n_iter: int = 10, p: int = 1
) -> (torch.Tensor):
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
        torch.Tensor: Returns optimal sampling plan.
    """
    q_arr:torch.Tensor = torch.tensor([1, 2, 5, 10, 20, 50, 100]).to(dev)  # as proposed in paper

    X3D:torch.Tensor = torch.zeros(len(q_arr), n, k, dtype=torch.float32).to(dev)
    X_start:torch.Tensor = rlh(n, k)

    for ix in range(7):
        X3D[ix,:,:] = evolve_lh(X_start, n_children, n_iter, q_arr[ix])

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(evolve_lh, X_start, n_children, n_iter, q_arr[ix]) for ix in range(7)]

    #     ix = 0
    #     for f in concurrent.futures.as_completed(results):    
    #         X3D[ix,:,:] = f.result()
    #         ix += 1

    indices:torch.Tensor = sort_morris_mitchel(X3D, p)

    X_best:torch.Tensor = X3D[indices[0]]

    print("Best LH found for q = {}.".format(q_arr[indices[0]]))

    return X_best