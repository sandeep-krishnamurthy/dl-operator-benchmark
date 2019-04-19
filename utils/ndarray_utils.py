import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from .profiler_utils import timer


@timer
def nd_forward_backward_and_time(F, runs, *args, **kwargs):
    """Helper function to run a given NDArray operator (F) for runs number of times with
    given args and kwargs. Executes both forward and backward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    :param F: NDArray operator (Function feference) to execute. Example: mx.nd.add
    :param runs: Number of time to execute the operation
    :param args: Arguments for the NDArray operator (F) being executed.
    :param kwargs: Key value arguments for the NDArray operator (F) being executed.
    :return: Total execution time in seconds, any results from NDArray operation execution.
    """
    for _ in range(runs):
        with mx.autograd.record():
            res = F(*args, **kwargs)
        res.backward()
        nd.waitall()


@timer
def nd_forward_and_time(F, runs, *args, **kwargs):
    """Helper function to run a given NDArray operator (F) for runs number of times with
    given args and kwargs. Executes ONLY forward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    :param F: NDArray operator (Function feference) to execute. Example: mx.nd.add
    :param runs: Number of time to execute the operation
    :param args: Arguments for the NDArray operator (F) being executed.
    :param kwargs: Key value arguments for the NDArray operator (F) being executed.
    :return: Total execution time in seconds, any results from NDArray operation execution.
    """
    for _ in range(runs):
        F(*args, **kwargs)
        nd.waitall()


def get_mx_ndarray(ctx, in_tensor, dtype, initializer, attach_grad=True):
    """Helper function to prepare a MXNet NDArray tensor in given Context (ctx) of type (dtype) with given
    initializer. You can get a new Tensor by providing only "Shape" or "Numpy NDArray" or another MXNet NDArray as
    "in_tensor".

    NOTE: This is a sync call and waits for the Tensor to be created.

    :param ctx: Context of the new MXNet NDArray Tensor.
    :param in_tensor: Can be a tuple of shape or Numpy NDArray or MXNet NDArray.
    :param dtype: Precision or Dtype of the expected Tensor. Ex: "float32", "Int64"
    :param initializer: Function reference to the initialize to use. Ex: mx.nd.random.normal, mx.nd.zeros
    :param attach_grad: To attach a gradient for the Tensor. Default is True.
    :return: MXNet NDArray Tensor.
    """
    if isinstance(in_tensor, tuple):
        tensor = initializer(ctx=ctx, shape=in_tensor, dtype=dtype)
    elif isinstance(in_tensor, np.ndarray):
        tensor = nd.array(in_tensor, ctx=ctx, dtype=dtype)
    elif isinstance(in_tensor, mx.ndarray):
        tensor = in_tensor.as_in_context(ctx=ctx).astype(dtype=dtype)

    else:
        raise ValueError("Invalid input type for creating input tensor. Input can be tuple() of shape or Numpy Array or"
                         " MXNet NDArray. Given - ", in_tensor)

    if attach_grad:
        tensor.attach_grad()

    tensor.wait_to_read()
    return tensor
