import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase

""" Performance benchmark tests for MXNet Gluon Pooling Layers
1. MaxPool1D
2. MaxPool2D
3. AvgPool1D
4. AvgPool2D
5. GlobalMaxPool1D
6. GlobalMaxPool2D
7. GlobalAvgPool1D
8. GlobalAvgPool2D
"""


class MaxPool1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon MaxPool1D Block.

    By default, benchmarks both forward and backward pass on the MaxPool1D block with pool_size 2, no strides,
    padding 0 with layout (N, C, W) on input of shape (32, 3, 256).

    This setting is influenced from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
                              "data_initializer": nd.normal,
                              "pool_size": 2,
                              "strides": None,
                              "padding": 0,
                              "layout": "NCW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.MaxPool1D(pool_size=self.inputs["pool_size"],
                                  strides=self.inputs["strides"],
                                  padding=self.inputs["padding"],
                                  layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_MaxPool1D_Forward_Backward_Time"] = exe_time / self.runs


class MaxPool2D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon MaxPool2D Block.

    By default, benchmarks both forward and backward pass on the MaxPool2D block with (2, 2) pool_size, no strides,
    (0, 0) padding with layout (N, C, H, W) on input of shape (32, 3, 256, 256).

    This setting is derived from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "pool_size": (2, 2),
                              "strides": None,
                              "padding": (0, 0),
                              "layout": "NCHW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.MaxPool2D(pool_size=self.inputs["pool_size"],
                                  strides=self.inputs["strides"],
                                  padding=self.inputs["padding"],
                                  layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_MaxPool2D_Forward_Backward_Time"] = exe_time / self.runs


class AvgPool1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon AvgPool1D Block.

    By default, benchmarks both forward and backward pass on the AvgPool1D block with pool_size 2, no strides,
    padding 0 with layout (N, C, W) on input of shape (32, 3, 256).

    This setting is influenced from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
                              "data_initializer": nd.normal,
                              "pool_size": 2,
                              "strides": None,
                              "padding": 0,
                              "layout": "NCW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.AvgPool1D(pool_size=self.inputs["pool_size"],
                                  strides=self.inputs["strides"],
                                  padding=self.inputs["padding"],
                                  layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_AvgPool1D_Forward_Backward_Time"] = exe_time / self.runs


class AvgPool2D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon AvgPool2D Block.

    By default, benchmarks both forward and backward pass on the AvgPool2D block with (2, 2) pool_size, no strides,
    (0, 0) padding with layout (N, C, H, W) on input of shape (32, 3, 256, 256).

    This setting is derived from ResNet architecture. By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "pool_size": (2, 2),
                              "strides": None,
                              "padding": (0, 0),
                              "layout": "NCHW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.AvgPool2D(pool_size=self.inputs["pool_size"],
                                  strides=self.inputs["strides"],
                                  padding=self.inputs["padding"],
                                  layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_AvgPool2D_Forward_Backward_Time"] = exe_time / self.runs


class GlobalMaxPool1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon GlobalMaxPool1D Block.

    By default, benchmarks both forward and backward pass on the GlobalMaxPool1D block with layout (N, C, W)
    on input of shape (32, 3, 256). By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
                              "data_initializer": nd.normal,
                              "layout": "NCW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.GlobalMaxPool1D(layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_GlobalMaxPool1D_Forward_Backward_Time"] = exe_time / self.runs


class GlobalMaxPool2D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon GlobalMaxPool1D Block.

    By default, benchmarks both forward and backward pass on the GlobalMaxPool1D block with layout (N, C, H, W)
    on input of shape (32, 3, 256, 256). By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "layout": "NCHW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.GlobalMaxPool2D(layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_GlobalMaxPool2D_Forward_Backward_Time"] = exe_time / self.runs


class GlobalAvgPool1D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon GlobalAvgPool1D Block.

    By default, benchmarks both forward and backward pass on the GlobalAvgPool1D block with layout (N, C, W)
    on input of shape (32, 3, 256). By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256),
                              "data_initializer": nd.normal,
                              "layout": "NCW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.GlobalAvgPool1D(layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_GlobalAvgPool1D_Forward_Backward_Time"] = exe_time / self.runs


class GlobalAvgPool2D(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon GlobalAvgPool2D Block.

    By default, benchmarks both forward and backward pass on the GlobalAvgPool2D block with layout (N, C, H, W)
    on input of shape (32, 3, 256, 256). By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=5, runs=25, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "layout": "NCHW",
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.GlobalAvgPool2D(layout=self.inputs["layout"])
        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_GlobalAvgPool2D_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_pooling_operations_benchmarks():
    """Helper to run all Gluon Pooling Layer benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    pooling_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls()
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        pooling_operations_results.append(benchmark_ref.get_benchmark_results())

    return pooling_operations_results
