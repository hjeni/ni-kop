import time


def measure_time(f):
    """
    Decorates the function with time measuring

    The time (in ns) is returned as the second parameter
    """

    def wrapper(*args, **kwargs):
        # start the clock
        time_init = time.perf_counter_ns()
        # call the function
        res = f(*args, **kwargs)
        # return the function result and the time of execution
        return res, time.perf_counter_ns() - time_init

    return wrapper


def divide_safe(a, b):
    """
    Computes a / b, return 0 for b = 0
    """
    return 0 if b == 0 else a / b


class FunctionCaller:
    """
    Simple wrapper around a function and its positional arguments

    Allows to call the function multiple times with the same arguments
    """
    def __init__(self, f, args):
        self.f = f
        self.args = args

    def call(self):
        return self.f(*self.args)


