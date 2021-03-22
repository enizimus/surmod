from typing import Callable, Tuple
import numpy as np
from ypstruct import structure


class Funzzy:
    def __init__(self: object, params: structure) -> None:

        self.params = params

        self._set_params()
        self._set_fuzzy_params()
        self._set_function()

    def _set_params(self):

        if type(self.params.p10) != list:
            self.params.p10 = [self.params.p10]

        if type(self.params.p90) != list:
            self.params.p90 = [self.params.p90]

        self.p10 = self.params.p10
        self.p90 = self.params.p90
        self.k = self.params.k
        self.fun_id = self.params.fun_id
        self.a = []
        self.b = []

    def _set_function(self):

        if self.fun_id == 1:
            self.f = lambda x: self._one_side_fun(x)
        elif self.fun_id == 2:
            self.f = lambda x: self._two_side_fun(x)
        elif self.fun_id == 3:
            self.f = lambda x: x

    def _set_fuzzy_params(self) -> None:

        for ind in range(self.fun_id):
            if self.fun_id < 3:
                self.a.append(
                    (self.p10[ind] * np.log(1 / 0.9 - 1) - self.p90[ind] * np.log(9))
                    / (np.log(1 / 0.9 - 1) - np.log(9))
                )
                self.b.append(-np.log(9) / (self.p10[ind] - self.a[ind]))
            else :
                self.a.append(0)
                self.b.append(0)

    def _one_side_fun(self, x: np.ndarray, ind: int = 0):
        return self.k / (1 + np.exp(-self.b[ind] * (x - self.a[ind])))

    def _two_side_fun(self, x):
        return (self._one_side_fun(x, 0) + self._one_side_fun(x, 1) - self.k)


class Fuzzificator:
    def __init__(self, Params: structure, method: str = "sum") -> None:

        self.funzzies = []

        for params in Params:
            self.funzzies.append(Funzzy(params))

        if method == "sum":
            self.f = lambda x: self._sum_fuzzies(x)
        elif method == "mult":
            self.f = lambda x: self._mult_fuzzies(x)
        elif method == "min":
            self.f = lambda x: self._min_fuzzies(x)

    def __call__(self, objectives: np.ndarray) -> float:

        return self.f(objectives)

    def _sum_fuzzies(self, objectives) -> float:

        quality = 0

        if len(objectives.shape) == 1:
            objectives = objectives[:,None]

        # print('________________________________')
        for ind, funzzy in enumerate(self.funzzies):
            curr_q = funzzy.f(objectives[:, ind])
            # print(f' Objective : {ind+1} : quality = {curr_q}')
            quality += curr_q
        # print('________________________________')
        return quality

    def _mult_fuzzies(self, objectives) -> float:

        quality = 1

        for funzzy, objective in zip(self.funzzies, objectives):
            quality *= funzzy.f(objective)

        return quality

    def _min_fuzzies(self, objectives) -> float:

        quality = np.inf

        for funzzy, objective in zip(self.funzzies, objectives):
            quality = min(funzzy.f(objective), quality)

        return quality
