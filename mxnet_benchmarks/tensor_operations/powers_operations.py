import mxnet as mx
import mxnet.ndarray as nd

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase

""" Performance benchmark tests for MXNet NDArray Powers operations
1. sqrt
2. square

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
3. Probably we can refactor the common logic of all these binary operations into a parent
   MXNetBinaryOperatorBenchmarkBase?
"""


class Sqrt(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Sqrt operation.

    By default benchmark both forward and backward element-wise Sqrt operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1024),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.sqrt, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.logical_not, runs=self.runs, data=self.data)

        self.results["MX_Sqrt_Forward_Backward_Time"] = exe_time / self.runs


class Square(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Square operation.

    By default benchmark both forward and backward element-wise Square operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1024),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.square, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.square, runs=self.runs, data=self.data)

        self.results["MX_Square_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_powers_operations_benchmarks(ctx, inputs):
    """Helper to run all Powers (sqrt, square) operator benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.


    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    power_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        power_operations_results.append(benchmark_ref.get_benchmark_results())

    return power_operations_results
