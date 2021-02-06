import numpy as np
import sys
from ypstruct import structure


def generate_random_variables(var_min: np.ndarray, var_max: np.ndarray):

    variables = np.random.uniform(size=(len(var_min),))
    variables = variables * (var_max - var_min) + var_min

    return variables


def crossover(par_1, par_2, gamma: float = 0.1):

    child_1, child_2 = par_1.deepcopy(), par_2.deepcopy()
    alpha = np.random.uniform(-gamma, 1 + gamma, *par_1.variables.shape)

    child_1.variables = alpha * par_1.variables + (1 - alpha) * par_2.variables
    child_2.variables = alpha * par_2.variables + (1 - alpha) * par_1.variables

    return child_1, child_2


def mutate(individual: object, mu: float, sigma: float):

    mutated = individual.deepcopy()
    mask = np.random.rand(*individual.variables.shape) <= mu

    indices = np.argwhere(mask)
    mutated.variables[indices] += sigma * np.random.randn(*indices.shape)

    return mutated


def check_bounds(individual: object, var_min: np.ndarray, var_max: np.ndarray):

    upper = var_max - individual.variables
    lower = individual.variables - var_min

    U = upper < 0
    L = lower < 0

    individual.variables[U] = var_max[U]
    individual.variables[L] = var_min[L]

    return individual


def set_optim_defaults(optim: object):

    keys = optim.keys()

    if not "var_min" in keys or not "var_max" in keys:
        sys.exit("Please give min and max values for optimization variables")

    if not "num_iter" in keys:
        optim.num_iter = 10

    if not "num_pop" in keys:
        optim.num_pop = 20

    if not "child_factor" in keys:
        optim.child_factor = 1

    if not "mu" in keys:
        optim.mu = 0.1

    if not "sigma" in keys:
        optim.sigma = 0.1

    if not "gamma" in keys:
        optim.gamma = 0.1

    return optim


def ga(ofun, optim: object, verbose: bool = True):

    optim = set_optim_defaults(optim)

    object_template = structure()
    object_template.variables = None
    object_template.cost = None

    best_individual = object_template.deepcopy()
    best_individual.cost = np.inf

    population = object_template.repeat(optim.num_pop)

    for individual in population:
        individual.variables = generate_random_variables(optim.var_min, optim.var_max)
        individual.cost = ofun(individual.variables)[0]

        if individual.cost < best_individual.cost:
            best_individual = individual.deepcopy()

    for iter in range(optim.num_iter):

        child_pop = []

        for _ in range(int(np.round(optim.child_factor * optim.num_pop / 2) * 2)):

            perm = np.random.permutation(optim.num_pop)
            child_1, child_2 = crossover(
                population[perm[0]], population[perm[1]], optim.gamma
            )

            child_1 = mutate(child_1, optim.mu, optim.sigma)
            child_2 = mutate(child_2, optim.mu, optim.sigma)

            child_1 = check_bounds(child_1, optim.var_min, optim.var_max)
            child_2 = check_bounds(child_2, optim.var_min, optim.var_max)

            child_1.cost = ofun(child_1.variables)[0]
            child_2.cost = ofun(child_2.variables)[0]

            if child_1.cost < best_individual.cost:
                best_individual = child_1.deepcopy()

            if child_2.cost < best_individual.cost:
                best_individual = child_2.deepcopy()

            child_pop += [child_1, child_2]

        population += child_pop
        population = sorted(population, key=lambda x: x.cost)
        population = population[: optim.num_pop]

        print(" >> Iter {} : cost = {} ".format(iter, best_individual.cost))
        print(" >> Params : {} \n".format(best_individual.variables))

    return best_individual.variables
