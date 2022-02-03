import pandas as pd
from itertools import count

from _utils import FunctionCaller, divide_safe, measure_time


class _ResultsTracker:
    """
    Tracks number of valid/invalid instances in the reference dataset
    """

    def __init__(self):
        self.t_sum, self.err_sum = 0, 0
        self.t_max, self.err_max = float('-inf'), float('-inf')

        self.n_valid, self.n_invalid = 0, 0
        self.n_solved, self.n_unsolved = 0, 0

    def add(self, t, err):
        """
        Adds new measuring
        """
        # error cannot be negative (discard)
        if err < 0 or t < 0:
            self.n_invalid += 1
            return
        # valid -> store time info
        self.n_valid += 1
        self.t_sum += t
        self.t_max = max(self.t_max, t)
        # error > 0 means negative objective (solution was not found)
        if err > 1:
            self.n_unsolved += 1
            return
        self.n_solved += 1
        # solved -> store error info
        self.err_sum += err
        self.err_max = max(self.err_max, err)

    def total_cnt(self):
        return self.n_valid + self.n_invalid

    def t_avg(self):
        return divide_safe(self.t_sum, self.n_valid)

    def err_avg(self):
        return divide_safe(self.err_sum, self.n_solved)

    def get_valid_ratio(self):
        total = self.total_cnt()
        return 1 if total == 0 else self.n_valid / total

    def debug_str(self):
        return f'[ValidityTracker]\n' \
               f'\tvalid - {self.n_valid} / {self.total_cnt()} ~ {self.get_valid_ratio() * 100:.2f} %\n' \
               f'\ttime (in s) -\tmax: {self.t_max / 10e9:.4f}, avg: {self.t_avg() / 10e9:.4f}\n' \
               f'\terror (rel) -\tmax: {self.err_max:.4f}, avg: {self.err_avg():.4f}\n'


# ----------------------------------------------------------------------------------------------------------------------


class PerformanceTracker:
    """
    Measures both time and error of a solver
    """

    def __init__(self, solver):
        self._solver = solver

        self._columns = ['file_id', 'time_max', 'time_avg', 'err_max', 'err_avg', 'n_instances', 'n_valid', 'n_solved']

    def go(self, inst_files_generator, sol_files_generator=None, indices=None, limit=None, verbose=False):
        """
        Returns pandas dataframe with solver's performance results
        """
        if indices is None:
            indices = count()

        if sol_files_generator is None:
            sol_files_generator = _gen_none()

        df = pd.DataFrame(columns=self._columns)
        # iterate top-level generator to get the low-level generators of actual instances and solutions
        for file_idx, inst_gen_f, sol_gen_f in zip(indices, inst_files_generator, sol_files_generator):
            # compare for given file
            rt = self._collect_results(inst_gen_f, sol_gen_f, limit)
            row = pd.Series((file_idx, rt.t_max, rt.t_avg(), rt.err_max, rt.err_avg(), rt.total_cnt(), rt.n_valid, rt.n_solved),
                            index=self._columns)
            df = df.append(row, ignore_index=True)

            if verbose:
                print(f'[{self._solver.__class__.__name__}]: {file_idx=}:\n{rt.debug_str()}\n')

        return df

    def _collect_results(self, instances_generator, solutions_generator=None, limit=None) -> _ResultsTracker:
        """
        Collects statistics over all passed instances and corresponding solutions
        """
        if solutions_generator is None:
            solutions_generator = _gen_none()

        rt = _ResultsTracker()
        # using counter instead of enumerate() to incorporate the limit option
        counter = count() if limit is None else range(limit)

        for _, inst, sol in zip(counter, instances_generator, solutions_generator):
            # get result (value) and measure time
            res, t = measure_time(self._solver.solve)(*inst)
            # compute error
            if sol is None:
                rt.add(t, 0)
            else:
                err_abs = sol.value - res
                err_rel = divide_safe(err_abs, sol.value)
                rt.add(t, err_rel)

        return rt


def _solve_with_solver(solver, tag, inst_files_generator_factory, sol_files_generator_factory, indices, limit, verbose):
    """
    Performs experiment with given solver (PerformanceTracker API wrapper)
    """
    # generators have to be created each time from scratch as they cannot be made into a copy
    inst_gen = inst_files_generator_factory.call()
    sol_gen = None if sol_files_generator_factory is None else sol_files_generator_factory.call()

    if verbose:
        print(f'[Solver: {tag}] Starting to measure performance..')

    # actual experiment with the solver
    pt = PerformanceTracker(solver)
    return pt.go(inst_gen, sol_gen, indices=indices, limit=limit, verbose=verbose)


def measure_solver(solver, inst_files_generator, sol_files_generator=None, indices=None, limit=None, verbose=False):
    """
    PerformanceTracker.go() wrapper

    Uses PerformanceTracker instance to collect stats about a solver
    """
    pt = PerformanceTracker(solver)
    return pt.go(inst_files_generator, sol_files_generator, indices, limit, verbose)


def measure_multiple_solvers(solvers_dict: dict,
                             inst_files_generator_factory: FunctionCaller,
                             sol_files_generator_factory: FunctionCaller = None,
                             indices=None,
                             limit=None,
                             verbose=False):
    """
    Requires dictionary of solvers in format "tag": <solver instance>

    Returns a dictionary in format "tag": <pandas DataFrame>
    """
    return {
        tag: _solve_with_solver(solver, tag, inst_files_generator_factory, sol_files_generator_factory, indices, limit, verbose)
        for tag, solver in solvers_dict.items()
    }


# infinite generator of 'None' values
def _gen_none():
    while True:
        yield None


