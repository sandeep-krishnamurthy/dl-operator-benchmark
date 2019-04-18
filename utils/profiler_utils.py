import functools
import time


def timer(func):
    """Decorator for timing the operation. Measures end to end execution time of the function in seconds.

    :param func: Operation to be executed and timed.
    :return: execution_time in seconds, Return value from operation
    """

    @functools.wraps(func)
    def timeit(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        exe_time = time.time() - start
        return exe_time, res

    return timeit
