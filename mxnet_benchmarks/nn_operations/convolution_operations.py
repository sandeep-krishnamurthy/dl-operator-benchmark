import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase

""" Performance benchmark tests for MXNet Gluon Convolution Layers
1. Conv1D
2. Conv2D
3. Conv1DTranspose
4. Conv2DTranspose (TODO: ON CPU)

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
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "channels": 64,
                              "kernel_size": (3, 3),
                              "strides": (1, 1),
                              "padding": (0, 0),
                              "dilation": (1, 1),
                              "layout": "NCHW",
                              "activation": None,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

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


class Conv2DTranspose(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Conv2DTranspose Block.

    By default, benchmarks both forward and backward pass on the Conv2DTranspose block with 64 channels, (3, 3) Kernel,
    (1, 1) strides, (0, 0) padding, (0,0) output_padding, (1, 1) dilation, with and input of shape (32, 3, 256, 256) and
    layout (N, C, H, W), without any activation function.

    This setting is derived from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "channels": 64,
                              "kernel_size": (3, 3),
                              "strides": (1, 1),
                              "padding": (0, 0),
                              "output_padding": (0, 0),
                              "dilation": (1, 1),
                              "layout": "NCHW",
                              "activation": None,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Conv2DTranspose(channels=self.inputs["channels"],
                                        kernel_size=self.inputs["kernel_size"],
                                        strides=self.inputs["strides"],
                                        padding=self.inputs["padding"],
                                        output_padding=self.inputs["output_padding"],
                                        dilation=self.inputs["dilation"],
                                        layout=self.inputs["layout"],
                                        activation=self.inputs["activation"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # TODO: Conv2DTranspose performance on CPU is real bad. Not recommended to run for now.
        if self.ctx == mx.cpu():
            print("WARNING: Conv2DTranspose performance on CPU is real bad. Not recommended to run for now!!")
            self.results["MX_Gluon_Imperative_Conv2DTranspose_Forward_Backward_Time"] = 0
            return

        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Conv2DTranspose_Forward_Backward_Time"] = exe_time / self.runs


class Conv1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Conv1D Block.

    By default, benchmarks both forward and backward pass on the Conv1D block with 64 channels, Kernel of size 3,
    stride of size 1, padding of size 0, dilation of size 1, with and input of shape (32, 3, 256) and layout (N, C, W),
    without any activation function.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
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

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

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


class Conv1DTranspose(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Conv1DTranspose Block.

    By default, benchmarks both forward and backward pass on the Conv1DTranspose block with 64 channels,
    Kernel of size 3, stride of size 1, padding of size 0, dilation of size 1, with and input of shape (32, 3, 256) and
    layout (N, C, W), without any activation function.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=1, runs=1, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
                              "data_initializer": nd.normal,
                              "channels": 64,
                              "kernel_size": 3,
                              "strides": 1,
                              "padding": 1,
                              "output_padding": 0,
                              "dilation": 1,
                              "layout": "NCW",
                              "activation": None,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Conv1DTranspose(channels=self.inputs["channels"],
                                        kernel_size=self.inputs["kernel_size"],
                                        strides=self.inputs["strides"],
                                        padding=self.inputs["padding"],
                                        output_padding=self.inputs["output_padding"],
                                        dilation=self.inputs["dilation"],
                                        layout=self.inputs["layout"],
                                        activation=self.inputs["activation"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Conv1DTranspose_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_convolution_operations_benchmarks(ctx, inputs):
    """Helper to run all Gluon Convolution Layer benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    nn_convolution_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        nn_convolution_operations_results.append(benchmark_ref.get_benchmark_results())

    return nn_convolution_operations_results
