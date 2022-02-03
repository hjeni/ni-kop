from abc import abstractmethod
import numpy as np
import copy

from solve.basic._solver import __Solver


class __SolverPrune(__Solver):
    """
    Brute force algorithm with possible pruning

    Template
    """

    def solve(self, n, cap, weights, values, target=None):
        self.set_instance_params(n, cap, weights, values, target)

        config_0 = [0] * n
        return self.__knapsack_rec(self._n, self._cap, 0, self._values_sum, config_0)

    def __knapsack_rec(self, n, cap, value, value_remaining, config):
        """
        Recursive function for B&B knapsack
        """
        # both B&B conditions
        if self._prune(cap, value, value_remaining):
            return 0

        # stop recurse
        if n == 0:
            # update best solution reached so far fo B&B pruning
            if value > self._value_best:
                self._value_best = value
                self._weight_best = self._cap - cap
                self._config_best = copy.deepcopy(config)
            return 0

        val_curr, weight_curr = self._values[n - 1], self._weights[n - 1]
        # try combination without current item
        config[n - 1] = 0
        without_curr = self.__knapsack_rec(n - 1, cap, value, value_remaining - self._values[n - 1], config)
        # current item cannot be used unless it fits into the knapsack
        if self._weights[n - 1] > cap:
            # (always better without)
            return without_curr
        # try combination with current item
        config[n - 1] = 1
        with_curr = self.__knapsack_rec(n - 1, cap - weight_curr, value + val_curr, value_remaining - val_curr, config)
        with_curr += self._values[n - 1]

        # return better option
        return max(with_curr, without_curr)

    @abstractmethod
    def _prune(self, cap, value, value_remaining):
        pass


class SolverBrute(__SolverPrune):
    """
    Brute-force knapsack solver
    """

    def _prune(self, cap, value, value_remaining):
        # never prune
        return False


class SolverBB(__SolverPrune):
    """
    Branch and bound knapsack solver
    """

    def _prune(self, cap, value, value_remaining):
        # two-directional pruning
        return cap < 0 or self._value_best > value + value_remaining


class SolverDynamic(__Solver):
    """
    Dynamic mem-table knapsack solver
    """

    def solve(self, n, cap, weights, values, target=None):
        self.set_instance_params(n, cap, weights, values, target)
        return self._solve_dynamic(n, cap, weights, values, target)

    def _solve_dynamic(self, n, cap, weights, values, target):
        """
        Dynamic mem-table base function

        Isolated from 'solve()' so that it can be used by any descendant
        """
        # init mem table
        n_rows, n_cols = self._values_sum + 1, n + 1
        table = [[float('inf')] * n_rows for _ in range(n_cols)]
        table[0][0] = 0
        # go through the mem table
        for i in range(1, n_cols):
            for j in range(n_rows):
                # set values with no items available to 0
                if j == 0:
                    table[i][j] = 0
                # for all other indices, step
                else:
                    if j - values[i - 1] < 0:
                        table[i][j] = table[i - 1][j]
                    else:
                        # take smaller from both options (with / without)
                        val_with = table[i - 1][j - values[i - 1]] + weights[i - 1]
                        val_without = table[i - 1][j]
                        table[i][j] = min(val_with, val_without)

        # find solution in the last column
        col = table[n]
        for i in range(n_rows - 1, 0, -1):
            if col[i] <= cap:
                self._value_best = i
                self._weight_best = col[i]
                self.__find_config(table, values)
                return i

        return 0

    def __find_config(self, table, values):
        """
        Finds the best item configuration assuming the mem-table is already filled
        """
        n = self._n
        val = self._value_best
        while n > 0:
            if table[n][val] != table[n - 1][val]:
                self._config_best[n - 1] = 1
                val -= values[n - 1]
            n -= 1
        pass


class SolverGreedy(__Solver):
    """
    Greedy knapsack solver
    """

    def solve(self, n, cap, weights, values, target=None):
        return self._solve_greedy(n, cap, weights, values, target)

    def _solve_greedy(self, n, cap, weights, values, target):
        """
        Greedy solver base function

        Isolated from 'solve()' so that it can be used by any descendant
        """
        self.set_instance_params(n, cap, weights, values, target)

        # use numpy so that argsort can be leveraged
        vw_ratios = np.array([v / w for v, w in zip(values, weights)])
        idx_permutation = np.argsort(-vw_ratios)

        cap_left = cap
        self.config_best = [0] * n
        # iterate all (sorted) objects, add them when they fit
        values_perm, weights_perm = np.array(values)[idx_permutation], np.array(weights)[idx_permutation]
        for i, (v, w) in enumerate(zip(values_perm, weights_perm)):
            if w <= cap_left:
                idx = idx_permutation[i]
                cap_left -= weights[idx]
                self._value_best += values[idx]
                self.config_best[idx] = 1

        self.weight_best = cap - cap_left
        return self._value_best


class SolverRedux(SolverGreedy):

    def solve(self, n, cap, weights, values, target=None):
        value_greedy = self._solve_greedy(n, cap, weights, values, target)
        value_solo, weight_solo, idx_solo = self._best_item(cap, weights, values)

        if value_solo > value_greedy:
            self._value_best = value_solo
            self._weight_best = weight_solo
            self._config_best = [0] * n
            self._config_best[idx_solo] = 1

        return self._value_best

    @staticmethod
    def _best_item(cap, weights, values):
        """
        Returns the most valuable item which fits into the knapsack
        """
        # filter out too heavy items (+ store indices)
        items = [(i, v, w) for i, (v, w) in enumerate(zip(values, weights)) if w <= cap]
        if len(items) == 0:
            return 0, 0, 0
        # find the pair with the biggest value
        idx, value, weight = max(items, key=lambda x: x[1])

        return value, weight, idx


class SolverFPTAS(SolverDynamic):

    def __init__(self, eps):
        super().__init__()
        self._eps = eps

    def solve(self, n, cap, weights, values, target=None):
        self.set_instance_params(n, cap, weights, values, target)

        # get rid of too heavy objects
        values_viable = [v for v, w in zip(values, weights) if w <= cap]
        if len(values_viable) == 0:
            return 0

        # compute new values, round to integers (dynamic programing cannot work with general floats)
        coef = (max(values_viable) * self._eps) / len(values_viable)
        values = [round(x / coef) for x in values]
        values_sum_orig = self._values_sum
        self._values_sum = sum(values)

        # use dynamic programming to find the result
        _ = self._solve_dynamic(n, cap, weights, values, target)

        # find the best value again from config, reset values sum
        self._value_best = sum([v for v, flag in zip(self._values, self._config_best) if flag == 1])
        self._values_sum = values_sum_orig

        return self._value_best














