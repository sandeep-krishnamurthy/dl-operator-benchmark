import mxnet as mx
import mxnet.ndarray as nd

from utils.profiler_utils import timer


@timer
def block_forward_backward_and_time(*args, block, runs, **kwargs):
    """Helper function to run a given Block (block) for 'runs' number of times with
    given args and kwargs. Executes both forward and backward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    :param block: Gluon block to execute. Example: an instance of gluon.nn.Dense(...)
    :param runs: Number of times to execute the block operation
    :param args: Arguments for the block being executed.
    :param kwargs: Key value arguments for the block being executed.
    :return: Tuple of (Total execution time in seconds, any results from block execution)
    """

    for _ in range(runs):
        with mx.autograd.record():
            res = block.forward(*args, **kwargs)
        res.backward()
        nd.waitall()
