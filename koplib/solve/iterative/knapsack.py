import random
import numpy as np

from solve.iterative._solver import SolverSA, _flip_random


class KnapsackSolverSA(SolverSA):
    """
    Simulated annealing solver for the knapsack problem
    """

    def __init__(self, cooling_coef=0.99,
                 t_initial=None,
                 t_final=None,
                 inner_cycle=1,
                 verbose=False,
                 start_acceptance_ratio=0.9,
                 sliding_window_size=25):
        """
        :param cooling_coef: SA cooling coefficient
        :param t_initial: SA initial temperature (estimated when not set)
        :param t_final: SA final temperature (when not set, the algo stops after declining all steps in a sliding window)
        :param inner_cycle: number of iterations with the same temperature
        :param verbose: important decisions are printed when switched on
        :param start_acceptance_ratio: only used to estimate initial temperature, target acceptance ratio to start with
        :param sliding_window_size: number of recent steps to focus on (when stopping, computing acceptance ratio etc.)
        """
        super().__init__(cooling_coef,
                         t_final,
                         t_initial,
                         inner_cycle,
                         verbose,
                         start_acceptance_ratio,
                         sliding_window_size)

        self._n, self._cap, self._weights, self._values = None, None, None, None

    def _init_problem(self, n, cap, weights, values):
        """
        Initializes the problem

        Returns initial configuration to begin with
        """
        self._n = n
        self._cap = cap
        self._weights = weights
        self._values = values

        # initial config with no items added
        return np.zeros(n)

    def _transition(self):
        """
        For current config, returns a random neighbour in the problem space
        """
        # add or remove an item
        return _flip_random(self._config)

    def _objective(self, config):
        """
        Scores a configuration (the higher, the better)
        """
        sum_w = sum([w for w, flag in zip(self._weights, config) if flag])
        # check for too heavy solutions
        if sum_w > self._cap:
            return self._cap - sum_w
        # return backpack value otherwise
        sum_v = sum([v for v, flag in zip(self._values, config) if flag])
        return sum_v


