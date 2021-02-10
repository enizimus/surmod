import numpy as np
from numpy.linalg import solve
from ypstruct import structure

from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.so_pso import PSO
from pymoo.algorithms.so_de import DE
from pymoo.optimize import minimize

class KrigOptim(Problem):

    def __init__(self, f, optim):
        n_var = len(optim.var_min) 
        xl = optim.var_min
        xu = optim.var_max 
        self.fun = f
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        
        y = np.zeros(x.shape[0])

        for ind in range(x.shape[0]):
            y[ind] = self.fun(x[ind][None,:])[0]

        out["F"] = y


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
        return np.dot(Phi, self.w)

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

        n, k = self.X.shape

        dX = np.zeros((k, n, n), dtype=np.float64)

        for ix in range(k):
            dX[ix, :, :] = self.X[:, ix].repeat(n).reshape(n, n)

        dX = np.linalg.norm(dX - np.transpose(dX, (0, 2, 1)), ord=2, axis=0)

        return self.basis_fun(dX)

    def __construct_gramm_mat_pred(self, X):

        n, k = self.X.shape
        l = X.shape[0]

        dX = np.zeros((k, l, n), dtype=np.float64)

        for ix in range(k):
            dX[ix, :, :] = self.X[:, ix].repeat(l).reshape(n, l).T - X[:, ix].repeat(
                n
            ).reshape(l, n)

        dX = np.linalg.norm(dX, ord=2, axis=0)

        return self.basis_fun(dX)

    def __estimate_weights(self):

        self.Phi = self.__construct_gramm_mat()

        self.w = np.linalg.solve(self.Phi, self.y)

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
        self:    object,
        optim:   object = None,
        verbose: bool   = False,
        infill:  bool   = False
    ):
        self.eps = 2.40e-16
        self.verbose = verbose
        self.optim = optim
        self.infill = infill
        self.n_feat = len(self.optim.var_min)//2

        if self.infill :
            print('Infill mode selected !')
            self.param_objective = lambda params: self.__infill_objective(params)
        else:
            self.param_objective = lambda params: self.__parameters_objective(params)

        self.krigopt = KrigOptim(self.param_objective, optim)
        self.algorithm = self.__get_optim_algo__()
        
        if self.verbose:
            print("Initialized Kriging object with : \n")
            self.__print_optim__()

    ##* User level functions (Public):

    def fit(self, X, y):

        self.X = X
        self.y = y

        res = minimize(self.krigopt, self.algorithm, ('n_gen', self.optim.num_iter), verbose=self.verbose)

        self.parameters = res.X[None,:]        
        self.Psi = self.param_objective(self.parameters)[1]

        if self.infill:
            return self.xopt
        else:
            return None

    def predict(self, X):

        self.theta = 10**self.parameters[0][: self.n_feat]
        self.p = self.parameters[0][self.n_feat :]

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
            "algorithm": self.optim.algorithm,
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
            algorithm=optim_dict["algorithm"]
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

    def __get_optim_algo__(self):

        if self.optim.algorithm.upper() == 'GA':
            algo = GA(pop_size=self.optim.num_pop, eliminate_duplicates=True) 
        elif self.optim.algorithm.upper() == 'DE':
            algo = DE(pop_size=self.optim.num_pop) 
        elif self.optim.algorithm.upper() == 'PSO':    
            algo = PSO(pop_size=self.optim.num_pop) 
        else :
            print('Not supported algorithm selected !')
            algo = []

        return algo

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

        n, k = self.X.shape
        Psi = np.zeros((k, n, n), dtype=np.float64)

        for ind in range(k):
            Psi[ind, :, :] = self.X[:, ind].repeat(n).reshape(n, n)

        Psi = np.abs(Psi - np.transpose(Psi, (0, 2, 1)))
        Theta = self.theta.repeat(n ** 2).reshape(k, n, n)
        P = self.p.repeat(n ** 2).reshape(k, n, n)

        return np.exp(-((Psi ** P) * Theta).sum(axis=0))

    def __construct_corr_mat_pred(self, X):

        n, k = self.X.shape
        l = X.shape[0]

        Psi = np.zeros((k, l, n), dtype=np.float64)

        for ix in range(k):
            Psi[ix, :, :] = np.abs(
                self.X[:, ix].repeat(l).reshape(n, l).T
                - X[:, ix].repeat(n).reshape(l, n)
            )

        Theta = self.theta.repeat(n * l).reshape(k, l, n)
        P = self.p.repeat(n * l).reshape(k, l, n)

        return np.exp(-((Psi ** P) * Theta).sum(axis=0))

    def __estimate_sig_mu_ln(self, Psi):

        n = self.X.shape[0]
        I = np.ones(n)
        mu = np.dot(I, solve(Psi, self.y)) / np.dot(I, solve(Psi, I))
        sigsq = np.dot((self.y - I * mu), solve(Psi, self.y - I * mu)) / n

        sigsq_log = 0 if sigsq == 0 else np.log(sigsq)
        psi_det = np.linalg.det(Psi)
        psidet_log = 0 if psi_det == 0 else np.log(psi_det)

        ln_like = -(-0.5 * n * sigsq_log - 0.5 * psidet_log)

        if ln_like == -np.inf:
            ln_like = np.inf

        return ln_like

    def __estimate_sig_mu_ln_infill(self, Psi, psi):

        n = self.X.shape[0]
        I = np.ones(n)

        mu = np.dot(I, solve(Psi, self.y)) / np.dot(I, solve(Psi, I))
        m  = (I*mu + psi*(self.optim.goal - mu)).ravel()
        C  = Psi - np.dot(psi, psi.T)

        sigsq = np.dot((self.y - m), solve(C, self.y - m)) / n

        sigsq_log = 0 if sigsq == 0 else np.log(sigsq)
        C_det = np.linalg.det(C)
        Cdet_log = 0 if C_det == 0 else np.log(C_det)

        ln_like = -(-0.5 * n * sigsq_log - 0.5 * Cdet_log)

        if ln_like == -np.inf:
            ln_like = np.inf

        return ln_like

    def __parameters_objective(self, parameters):

        #parameters = parameters*(self.optim.var_max-self.optim.var_min) + self.optim.var_min

        self.theta = 10**parameters[0][: self.n_feat]
        self.p = parameters[0][self.n_feat :]

        Psi = self.__construct_corr_mat()
        ln_like = self.__estimate_sig_mu_ln(Psi)

        return ln_like, Psi

    def __infill_objective(self, parameters):

        self.theta = 10 ** parameters[0][: self.n_feat]
        self.p = parameters[0][self.n_feat:self.n_feat*2]
        self.xopt = parameters[0][self.n_feat*2:]

        Psi = self.__construct_corr_mat()
        psi = self.__construct_corr_mat_pred(self.xopt[:,None])

        ln_like = self.__estimate_sig_mu_ln_infill(Psi, psi)

        return ln_like, Psi

# import matplotlib.pyplot as plt

# X = np.linspace(0,2*np.pi,5)[:,None]
# y = np.sin(X)

# varmin = np.array([0.01, 1])
# varmax = np.array([10, 3])
# numiter = 5
# numpop = 50
# mu = 0.1
# sigma = 0.2
# child_factor = 2
# gamma = 0.1

# optim = structure(var_min=varmin, var_max=varmax, num_iter=numiter, num_pop=numpop, mu=mu, sigma=sigma, child_factor=2,  gamma=gamma)
# krigger = Kriging(optim, verbose=True)
# krigger.fit(X, y.ravel())

# X_test = np.linspace(0,2*np.pi,50)[:,None] #np.array([[np.pi/4+np.pi/10, np.pi+np.pi/11]])

# y_hat = krigger.predict(X_test)

# print(krigger.theta)

# fig, ax = plt.subplots()
# ax.scatter(X, y)
# ax.plot(X_test, y_hat)
# plt.show()