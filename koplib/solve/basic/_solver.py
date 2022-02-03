from abc import ABC, abstractmethod


class __Solver(ABC):
    """
    Represents any knapsack solver
    """

    def __init__(self):
        self._n = None
        self._cap = None
        self._weights = None
        self._values = None
        self._target = None
        self._is_constructive = None
        self._value_best = None
        self._weight_best = None
        self._config_best = None
        self._values_sum = None

    def set_instance_params(self, n, cap, weights, values, target):
        self._n = n
        self._cap = cap
        self._weights = weights
        self._values = values
        self._target = target
        self._is_constructive = target is None
        # reset best configuration found
        self._value_best = 0
        self._weight_best = 0
        self._config_best = [0] * n
        # compute max possible value of the knapsack
        self._values_sum = sum(self._values)

    def is_solvable(self):
        return max(self._config_best) > 0

    def get_config(self):
        return self._config_best

    @abstractmethod
    def solve(self, n, cap, weights, values, target=None):
        pass

