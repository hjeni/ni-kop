import random
import numpy as np

from solve.iterative._solver import SolverSA, _flip_random


def _eval_clause(clause, config):
    """
    Evaluates given SAT clause with either 1 or 0
    """
    indices = [abs(x) for x in clause]
    # get assignments of the variables used and corresponding signs from the clause
    assigns = [config[idx - 1] for idx in indices]
    signs = [0 if x < 0 else 1 for x in clause]

    # either both have to be negative or positive at 1+ variables to evaluate the clause as correct
    return 1 if sum([a == b for a, b in zip(assigns, signs)]) > 0 else 0


class WSATSolverSA(SolverSA):
    """
    Simulated annealing solver for the weighted SAT problem
    """

    def __init__(self, cooling_coef=0.99,
                 t_initial=None,
                 t_final=None,
                 inner_cycle=1,
                 verbose=False,
                 start_acceptance_ratio=0.9,
                 sliding_window_size=25,
                 penalize_strategy='simple',
                 weights_importance=0.2):
        """
        :param cooling_coef: SA cooling coefficient
        :param t_initial: SA initial temperature (estimated when not set)
        :param t_final: SA final temperature (when not set, the algo stops after declining all steps in a sliding window)
        :param inner_cycle: number of iterations with the same temperature
        :param verbose: important decisions are printed when switched on
        :param start_acceptance_ratio: only used to estimate initial temperature, target acceptance ratio to start with
        :param sliding_window_size: number of recent steps to focus on (when stopping, computing acceptance ratio etc.)
        :param penalize_strategy: objective strategy to penalize incorrect solutions ('simple', 'weighted', 'greedy')
        :param weights_importance: only used for greedy strategy (the higher, the greedier ~ <0, 1>)
        """
        super().__init__(cooling_coef,
                         t_final,
                         t_initial,
                         inner_cycle,
                         verbose,
                         start_acceptance_ratio,
                         sliding_window_size)

        self._n_vars, self._n_clauses, self._weights, self._clauses = None, None, None, None

        assert penalize_strategy in ['simple', 'weighted', 'greedy'], 'Undefined penalization strategy!'
        self._penalize = penalize_strategy
        self._weights_importance = weights_importance

    def _init_problem(self, n_vars, n_forms, weights, clauses) -> np.array:
        """
        Initializes the problem

        Returns initial configuration to begin with
        """
        self._n_vars = n_vars
        self._n_clauses = n_forms
        self._weights = weights
        self._clauses = clauses

        self._max_score = sum(weights)

        return np.zeros(n_vars)

    def _transition(self):
        """
        For current config, returns a random neighbour in the problem space
        """
        # switch one variable
        return _flip_random(self._config)

    def _objective(self, config):
        """
        Scores a configuration (the higher, the better)
        """
        eval_vec = [_eval_clause(c, config) for c in self._clauses]
        weighted_sum = sum([w * x for w, x in zip(self._weights, config)])
        n_invalid = self._n_clauses - sum(eval_vec)
        # in case all clauses are correct, return the weighted sum of active variables
        if n_invalid == 0:
            return weighted_sum

        if self._penalize == 'simple':
            pen = n_invalid
        elif self._penalize == 'weighted':
            # scale up to max score
            invalid_ratio = n_invalid / self._n_clauses
            pen = invalid_ratio * self._max_score
        elif self._penalize == 'greedy':
            try:
                # support higher weighted sums
                score_ratio = (self._max_score - weighted_sum) / self._max_score * self._weights_importance
                # penalize unsatisfied clauses
                invalid_ratio = n_invalid / self._n_clauses * (1 - self._weights_importance)
                pen = self._max_score * (invalid_ratio + score_ratio)
            # just in case (self._max_score == 0)
            except ZeroDivisionError:
                pen = n_invalid
        else:
            pen = 0

        return - pen




