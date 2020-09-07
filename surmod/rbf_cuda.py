import numpy as np
from numpy.linalg import solve
import torch
import matplotlib.pyplot as plt
from ypstruct import structure

from . import genetic_algorithm


class RBF:
    """Radial Basis Functions (RBF) Class :
    - Objects of this class implement a *fit()* function for fitting
    the RBF model to the data and the *predict()* function for running
    the predictions of the model on the input data.
    """

    def __init__(
        self: object,
        basis: str = "cubic",
        sigma: float = 0.1,
        sigma_int: list = [-2, 2],
        n_sigma: int = 20,
        verbose: bool = False,
    ):
        self.sigma = sigma
        self.sigma_int = sigma_int
        self.sigma_arr = np.logspace(sigma_int[0], sigma_int[1], n_sigma)
        self.sigma_best = np.nan
        self.n_sigma = n_sigma
        self.basis = basis
        self.basis_fun, self.basis_num = self.__get_basis_function(basis)
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print("Initialized RBF object with : ")
            print(" - Basis : {}".format(self.basis))
            print(" - Sigma : {}".format(self.sigma))
            print(" - Sigma interval : {}".format(self.sigma_int))
            print(" - Num Sigmas : {}".format(self.n_sigma))

    ##* User level functions (Public):

    def fit(self, X, y):

        self.X = X
        self.y = y
        self.__estimate_weights()

    def predict(self, X):
        Phi = self.__construct_gramm_mat_pred(X)
        return torch.mm(Phi, self.w)

    def get_params(self):
        return self.sigma_arr

    def set_params(self, sigma):
        self.sigma = sigma

    def save(self, path: str = "./rbf_model.npy"):

        model = {
            "sigma": self.sigma,
            "sigma_int": self.sigma_int,
            "basis": self.basis,
            "n_sigma": self.n_sigma,
            "w": self.w,
            "X": self.X,
        }

        np.save(path, model, allow_pickle=True)

    def load(self, path: str = "./rbf_model.npy"):

        if path[-4:] != ".npy":
            path += ".npy"

        model = np.load(path, allow_pickle=True)[()]

        self.X = model["X"]
        self.w = model["w"]
        self.sigma = model["sigma"]
        self.sigma_int = model["sigma_int"]
        self.basis = model["basis"]
        self.n_sigma = model["n_sigma"]
        self.sigma_arr = np.logspace(self.sigma_int[0], self.sigma_int[1], self.n_sigma)
        self.basis_fun, self.basis_num = self.__get_basis_function(self.basis)

        print("Loaded parameters from saved model :\n {}".format(path))

    ## -------------------------------------------
    ##* Dev level functions (Private):

    def __construct_gramm_mat(self):

        n = self.X.shape[0]
        dX = torch.zeros(n, n)

        for ix in range(n):
            for iy in range(ix):
                dX[ix, iy] = torch.norm(self.X[ix, :] - self.X[iy, :], 2)
                dX[iy, ix] = dX[ix, iy]

        return self.basis_fun(dX)

    def __construct_gramm_mat_pred(self, X):

        n = self.X.shape[0]
        k = X.shape[0]

        dX = torch.zeros(k, n)

        for ix in range(k):
            for iy in range(n):
                dX[ix, iy] = torch.norm(X[ix, :] - self.X[iy, :], 2)

        return self.basis_fun(dX)

    def __estimate_weights(self):

        Phi = self.__construct_gramm_mat()

        self.w = torch.solve(self.y, Phi)

    def __basis_linear(self, r):
        return r

    def __basis_cubic(self, r):
        return r ** 3

    def __basis_thin_plate_spline(self, r):
        I = r == 0
        res = np.zeros_like(r)
        res[~I] = r[~I] ** 2 * np.log(r[~I])
        return res

    def __basis_gaussian(self, r):
        return np.exp(-(r ** 2) / (2 * self.sigma ** 2))

    def __basis_multiquadric(self, r):
        return (r ** 2 + self.sigma ** 2) ** 0.5

    def __basis_inv_multiquadric(self, r):
        return (r ** 2 + self.sigma ** 2) ** (-0.5)

    def __get_basis_function(self, basis: str):

        basis_functions = {
            "linear": (self.__basis_linear, 1),
            "thin_plate_spline": (self.__basis_thin_plate_spline, 2),
            "cubic": (self.__basis_cubic, 3),
            "gaussian": (self.__basis_gaussian, 4),
            "multiquadric": (self.__basis_multiquadric, 5),
            "inverse_multiquadric": (self.__basis_inv_multiquadric, 6),
        }

        return basis_functions[basis]


class Kriging:
    def __init__(
        self: object,
        optim: object = None,
        verbose: bool = False,
    ):
        self.eps = 2.22e-16
        self.verbose = verbose
        self.optim = optim

        if self.verbose:
            print("Initialized Kriging object with : \n")
            self.__print_optim__()

    ##* User level functions (Public):

    def fit(self, X, y):

        self.X = X
        self.y = y

        self.n_feat = X.shape[1]

        self.parameters = genetic_algorithm.ga(self.__parameters_objective, self.optim)
        self.Psi = self.__parameters_objective(self.parameters)[1]

    def predict(self, X):

        self.theta = 10 ** self.parameters[: self.n_feat]
        self.p = self.parameters[self.n_feat :]

        I = np.ones(self.X.shape[0])

        mu = np.dot(I, solve(self.Psi, self.y)) / np.dot(I, solve(self.Psi, I))

        Psi = self.__construct_corr_mat_pred(X)

        y_hat = mu + np.dot(Psi, solve(self.Psi, self.y - I * mu))

        return y_hat

    def save(self, path: str = "./kriging_model.npy"):

        optim = {
            "var_min": self.optim.var_min,
            "var_max": self.optim.var_max,
            "num_iter": self.optim.num_iter,
            "num_pop": self.optim.num_pop,
            "mu": self.optim.mu,
            "sigma": self.optim.sigma,
            "child_factor": self.optim.child_factor,
            "gamma": self.optim.gamma,
        }

        model = {
            "parameters": self.parameters,
            "X": self.X,
            "n_feat": self.n_feat,
            "Psi": self.Psi,
            "y": self.y,
            "optim": optim,
        }

        np.save(path, model, allow_pickle=True)

    def load(self, path: str = "./kriging_model.npy"):

        if path[-4:] != ".npy":
            path += ".npy"

        model = np.load(path, allow_pickle=True)[()]
        optim_dict = model["optim"]

        optim = structure(
            var_min=optim_dict["var_min"],
            var_max=optim_dict["var_max"],
            num_iter=optim_dict["num_iter"],
            num_pop=optim_dict["num_pop"],
            mu=optim_dict["mu"],
            sigma=optim_dict["sigma"],
            child_factor=optim_dict["child_factor"],
            gamma=optim_dict["gamma"],
        )

        self.parameters = model["parameters"]
        self.X = model["X"]
        self.n_feat = model["n_feat"]
        self.Psi = model["Psi"]
        self.y = model["y"]
        self.optim = optim

        print("Loaded parameters from saved model :\n {}".format(path))

    ## -------------------------------------------
    ##* Dev level functions (Private):

    def __print_optim__(self):
        
        print("Optimization parameters : ")
        print("  var_min : ", self.optim.var_min)
        print("  var_max : ", self.optim.var_max)
        print("  num_iter : ", self.optim.num_iter)
        print("  num_pop : ", self.optim.num_pop)
        print("  child_factor : ", self.optim.child_factor)
        print("  mu : ", self.optim.mu)
        print("  sigma : ", self.optim.sigma)
        print("  gamma : ", self.optim.gamma)
        

    def __construct_corr_mat(self):

        n = self.X.shape[0]
        Psi = np.zeros((n, n))

        for ix in range(n):
            for iy in range(ix):
                Psi[ix, iy] = self.__basis(self.X[ix, :], self.X[iy, :])

        Psi = Psi + Psi.T + np.eye(n) + np.eye(n) * self.eps

        return Psi

    def __construct_corr_mat_pred(self, X):

        n = self.X.shape[0]
        k = X.shape[0]
        Psi = np.zeros((k, n))

        for ix in range(n):
            for iy in range(k):
                Psi[iy, ix] = self.__basis(self.X[ix, :], X[iy, :])

        # Psi = Psi + Psi.T + np.eye(n) + np.eye(n) * self.eps

        return Psi

    def __estimate_sig_mu_ln(self, Psi):

        n = self.X.shape[0]
        I = np.ones(n)
        mu = np.dot(I, solve(Psi, self.y)) / np.dot(I, solve(Psi, I))
        sigsq = np.dot((self.y - I * mu), solve(Psi, self.y - I * mu)) / n

        ln_like = -(-0.5 * n * np.log(sigsq) - 0.5 * np.log(np.linalg.det(Psi)))

        if ln_like == -np.inf:
            ln_like = np.inf

        return ln_like

    def __parameters_objective(self, parameters):

        self.theta = 10 ** parameters[: self.n_feat]
        self.p = parameters[self.n_feat :]

        Psi = self.__construct_corr_mat()
        ln_like = self.__estimate_sig_mu_ln(Psi)
        return ln_like, Psi

    def __basis(self, x1, x2):
        return np.exp(-np.dot(self.theta, np.abs(x1 - x2) ** self.p))