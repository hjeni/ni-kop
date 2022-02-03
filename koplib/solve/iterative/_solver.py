import math
import random
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
import matplotlib.pyplot as plt


def _with_prob(p):
    r = random.random()
    return r <= p


def _flip_random(config: np.array):
    """
    Flips random binary element of the array
    """
    idx = random.randint(0, len(config) - 1)
    new_config = config.copy()
    new_config[idx] = (config[idx] + 1) % 2
    return new_config


class SolverSA(ABC):
    """
    Simulated annealing solver template
    """

    StepResult = namedtuple('StepResult', ['objective', 'accepted'])

    def __init__(self, cooling_coef=0.99,
                 t_initial=None,
                 t_final=None,
                 inner_cycle=1,
                 verbose=False,
                 start_acceptance_ratio=0.9,
                 sliding_window_size=30):
        """
        :param cooling_coef: SA cooling coefficient
        :param t_initial: SA initial temperature (estimated when not set)
        :param t_final: SA final temperature (when not set, the algo stops after declining all steps in a sliding window)
        :param inner_cycle: number of iterations with the same temperature
        :param verbose: important decisions are printed when switched on
        :param start_acceptance_ratio: only used to estimate initial temperature, target acceptance ratio to start with
        :param sliding_window_size: number of recent steps to focus on (when stopping, computing acceptance ratio etc.)
        """
        self._t_final = t_final
        self._cooling_coef = cooling_coef
        self._t_initial = t_initial
        self._inner_cycle = inner_cycle
        self._verbose = verbose
        self._start_acceptance_ratio = start_acceptance_ratio
        self._sliding_window_size = sliding_window_size

        self._t_curr = self._t_initial
        self._config, self._history, self._n_steps_still = None, None, None

    def _estimate_initial_temp(self):
        n_samples = self._sliding_window_size
        # start with big number
        estimate = 1e7 * random.uniform(1, 2)

        # keep lowering the initial temperature until the acceptance ratio declines enough
        while estimate > 0:
            self._t_curr = estimate
            n_accepted = 0
            for _ in range(n_samples):
                self._config, _ = self._step()
                if self._n_steps_still == 0:
                    n_accepted += 1
            # check acceptance ratio
            if n_accepted / n_samples < self._start_acceptance_ratio:
                if self._verbose:
                    print(f'[SA solver] Initial temperature estimate: {estimate}')
                return estimate
            # lower the estimate
            estimate /= 2
        return 0

    def solve(self, *args):
        self._n_steps_still = 0
        config_init = self._init_problem(*args)
        self._config = config_init
        self._history = []
        # estimate initial temperature when it's not hardcoded
        self._t_curr = self._estimate_initial_temp() if self._t_initial is None else self._t_initial
        self._config = config_init

        # repeat until the final temp is reached
        while not self._stop():
            # inner cycle with stable temperature
            for _ in range(self._inner_cycle):
                config_next, objective_next = self._step()
                self._config = config_next
                step_result = self.StepResult(objective=objective_next, accepted=(self._n_steps_still == 0))
                self._history.append(step_result)
            # cool down
            self._t_curr *= self._cooling_coef

        obj = self._objective(self._config)
        if self._verbose:
            print(f'[SA finished] objective: {obj}, temperature: {self._t_curr}')
        return obj

    def _stop(self):
        """
        Evaluates whether the algorithm converged or not
        """
        # either the final temperature has been reached or the configuration has not changed for long
        converged = self._n_steps_still >= self._sliding_window_size / 2 if self._t_final is None else self._t_curr <= self._t_final
        return converged or self._t_curr <= 0

    def _step(self):
        """
        Performs one step

        Returns a tuple: (new config, objective, flag)

        new config: new configuration to step to
        objective: objective score of the new configuration
        flag: True when new configuration was accepted, False otherwise
        """
        # try to move from current state
        neigh = self._transition()
        obj_new, obj_curr = self._objective(neigh), self._objective(self._config)
        improvement = obj_new - obj_curr

        # assume neighbour gets accepted
        config_next, objective_next, change = neigh, obj_new, True
        # decline worse neighbours based on the current temperature
        if improvement < 0:
            p = math.exp(improvement / self._t_curr)
            if not _with_prob(p):
                # change new config & objective back to original values
                config_next = self._config
                change = False
        self._n_steps_still = 0 if change else self._n_steps_still + 1
        return config_next, objective_next

    def get_config(self):
        """
        Returns the last configuration found
        """
        return self._config

    def get_temperature(self):
        """
        Returns the last final temperature
        """
        return self._t_curr

    def get_objective(self):
        """
        Returns the last objective score reached
        """
        return self._objective(self._config)

    def history(self):
        """
        Returns history of objective function value in each step of the previous run
        """
        return self._history

    def get_n_steps_still(self):
        """
        Returns number of steps declined in a row (indicates convergence)
        """
        return self._n_steps_still

    def inspect(self):
        """
        Prints out info about a run of the last solving process
        """
        print(f'final temperature: {self.get_temperature():.4f}')
        print(f'number of steps unchanged: {int(self.get_n_steps_still())}')
        print(f'final score: {self.get_objective():.2f}')
        h_obj = [x.objective for x in self.history()]
        plt.plot(h_obj)
        plt.show()

    @abstractmethod
    def _init_problem(self, *args) -> np.array:
        """
        Initializes the problem

        Returns initial configuration to begin with
        """
        pass

    @abstractmethod
    def _transition(self):
        """
        For current config, returns a random neighbour in the problem space
        """
        pass

    @abstractmethod
    def _objective(self, config):
        """
        Scores a configuration (the higher, the better)
        """
        pass

