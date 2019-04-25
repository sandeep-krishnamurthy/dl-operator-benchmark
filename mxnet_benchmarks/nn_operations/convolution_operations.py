import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase

""" Performance benchmark tests for MXNet Gluon Convolution Layers
1. Conv1D
2. Conv2D
3. Conv1DTranspose
4. Conv2DTranspose

NOTE: Number of warmup and benchmark runs for convolution is reduced as the computation is heavy and within
first 25 runs results stabilizes without variation.
"""


class Conv2D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Conv2D Block.

    By default, benchmarks both forward and backward pass on the Conv2D block with 64 channels, (3, 3) Kernel,
    (1, 1) strides, (0, 0) padding, (1, 1) dilation, with and input of shape (32, 3, 256, 256) and layout (N, C, H, W),
    without any activation function.

    This setting is derived from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (32, 3, 256, 256),
                      "data_initializer": nd.normal,
                      "channels": 64,
                      "kernel_size": (3, 3),
                      "strides": (1, 1),
                      "padding": (1, 1),
                      "dilation": (1, 1),
                      "layout": "NCHW",
                      "activation": None,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Conv2D(channels=self.inputs["channels"],
                               kernel_size=self.inputs["kernel_size"],
                               strides=self.inputs["strides"],
                               padding=self.inputs["padding"],
                               dilation=self.inputs["dilation"],
                               layout=self.inputs["layout"],
                               activation=self.inputs["activation"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Conv2D_Forward_Backward_Time"] = exe_time / self.runs


class Conv1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Conv1D Block.

    By default, benchmarks both forward and backward pass on the Conv1D block with 64 channels, Kernel of size 3,
    stride of size 1, padding of size 0, dilation of size 1, with and input of shape (32, 3, 256) and layout (N, C, W),
    without any activation function.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (32, 3, 256),
                      "data_initializer": nd.normal,
                      "channels": 64,
                      "kernel_size": 3,
                      "strides": 1,
                      "padding": 1,
                      "dilation": 1,
                      "layout": "NCW",
                      "activation": None,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Conv1D(channels=self.inputs["channels"],
                               kernel_size=self.inputs["kernel_size"],
                               strides=self.inputs["strides"],
                               padding=self.inputs["padding"],
                               dilation=self.inputs["dilation"],
                               layout=self.inputs["layout"],
                               activation=self.inputs["activation"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Conv1D_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_convolution_operations_benchmarks():
    """Helper to run all Gluon Convolution Layer benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """

    benchmark_ref = Conv2D()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = Conv1D()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()


run_all_gluon_nn_convolution_operations_benchmarks()
