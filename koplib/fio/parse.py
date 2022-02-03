from abc import abstractmethod, ABC

from collections import namedtuple
import numpy as np

from _utils import FunctionCaller


ReferenceSolution = namedtuple('ReferenceSolution', ['value', 'config'])


class __Parser(ABC):
    """
    Represents any input file parser

    Assumes each item is represented by one line of given file
    """

    def __init__(self):
        self._path = None
        self._generator = None

    def generate_all(self):
        """
        Returns a generator of all remaining items
        """
        return self._generator

    def init(self, path):
        """
        Initializes items generator
        """
        self._path = path
        self._generator = self._get_new_generator()

    def get_next(self):
        """
        Returns next item and moves the cursor
        """
        if self._generator is None:
            return None
        return next(self._generator)

    def _get_new_generator(self):
        """
        Returns a generator of all items in the file
        """
        with open(self._path, "r") as f:
            for line in f:
                yield self.parse_line(line)

    @abstractmethod
    def parse_line(self, s: str):
        """
        Defines how to parse the items
        """
        pass


# ----------------------------------------------------------------------------------------------------------------------


class ParserKnapsackInst(__Parser):
    """
    Parses knapsack problem instances
    """

    def parse_line(self, s: str):
        split = s.split()
        rid, n, cap = [int(x) for x in split[:3]]
        weights, values = [int(x) for x in split[3::2]], [int(x) for x in split[4::2]]

        return n, cap, weights, values


class ParserKnapsackSol(__Parser):
    """
    Parses knapsack problem solutions
    """

    def parse_line(self, s: str):
        split = s.split()
        rid, n, val_ref = [int(x) for x in split[:3]]
        res_ref = np.array([int(x) for x in split[3:]])

        return ReferenceSolution(val_ref, res_ref)


class ParserWSATInst(__Parser):
    """
    Parses weighted 3-SAT problem instances
    """

    def parse_line(self, s: str):
        split = s.split()
        n_vars, n_clauses = [int(x) for x in split[:2]]
        clauses_start = (n_vars + 2)
        weights = [int(x) for x in split[2:clauses_start]]
        # clauses (more complicated, they are tuples)
        clauses = []
        start = clauses_start
        for _ in range(n_clauses):
            end = start + 3
            clauses.append(tuple(int(x) for x in split[start:end]))
            start = end

        return n_vars, n_clauses, weights, clauses


class ParserWSATSol(__Parser):
    """
    Parses weighted SAT problem solutions
    """

    def parse_line(self, s: str):
        split = s.split()
        # format: <instance ID> <optimal value> <config[]> <ending zero>
        value = int(split[1])
        config = np.array([1 if int(x) > 0 else 0 for x in split[2:-1]])
        return ReferenceSolution(value, config)


# ----------------------------------------------------------------------------------------------------------------------

def parse_dataset(file_paths, parser_class):
    """
    Returns a generator (1 element per file) of file items generators (1 element per row)
    """
    # strategy pattern using the parser class
    parser = parser_class()
    for path in file_paths:
        parser.init(path)
        yield parser.generate_all()


def create_data_gen_factory(file_paths, parser_class):
    """
    parse_dataset wrapper using _utils.objects.FunctionCaller

    Returns a callable data generator getter
    """
    return FunctionCaller(f=parse_dataset, args=[file_paths, parser_class])



