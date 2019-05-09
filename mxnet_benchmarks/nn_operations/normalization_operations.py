import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time

""" Performance benchmark tests for MXNet Gluon Normalization Layers
1. Dropout
2. BatchNorm
"""


class Dropout(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Dropout Block.

    By default, benchmarks both forward and backward pass on the Dropout block using an input tensor of shape
    (32, 3, 256, 256) using 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) with rate=0.5
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "rate": 0.5,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Dropout(rate=self.inputs["rate"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Dropout_Forward_Backward_Time"] = exe_time / self.runs


class BatchNorm(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon BatchNorm Block.

    By default, benchmarks both forward and backward pass on the BatchNorm block using an input tensor of shape
    (32, 3, 256, 256) using 'float32' precision. Uses default momentum(0.9), epsilon(1e-05) with center and scale
    set to True.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256)
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.BatchNorm()

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_BatchNorm_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_normalization_operations_benchmarks(ctx, inputs):
    """Helper to run all Gluon Normalization Layer benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    normalization_operation_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        normalization_operation_results.append(benchmark_ref.get_benchmark_results())

    return normalization_operation_results
