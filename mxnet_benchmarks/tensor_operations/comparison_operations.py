import mxnet as mx
import mxnet.ndarray as nd

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

"""Performance benchmark tests for MXNet NDArray Comparison Operations
1. lesser
2. lesser_equal
3. greater
4. greater_equal
5. equal
6. not_equal

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
3. Probably we can refactor the common logic of all these binary operations into a parent
   MXNetBinaryOperatorBenchmarkBase?
"""


class Lesser(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Lesser operation.

    By default benchmark both forward and backward element_wise Lesser operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.lesser, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.lesser, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Lesser_Forward_Backward_Time"] = exe_time / self.runs


class LesserEqual(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor LesserEqual operation.

    By default benchmark both forward and backward element_wise LesserEqual operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.lesser_equal, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.lesser_equal, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Lesser_Equal_Forward_Backward_Time"] = exe_time / self.runs


class Greater(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Greater operation.

    By default benchmark both forward and backward element_wise Greater operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.greater, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.greater, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Greater_Forward_Backward_Time"] = exe_time / self.runs


class GreaterEqual(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor GreaterEqual operation.

    By default benchmark both forward and backward element_wise GreaterEqual operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.greater_equal, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.greater_equal, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Greater_Equal_Forward_Backward_Time"] = exe_time / self.runs


class Equal(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Equal operation.

    By default benchmark both forward and backward element_wise Equal operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.equal, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.equal, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Equal_Forward_Backward_Time"] = exe_time / self.runs


class NotEqual(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Not_Equal operation.

    By default benchmark both forward and backward element_wise Not_Equal operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"lhs": (1024, 1024),
                              "rhs": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.lhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["lhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])
        self.rhs = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["rhs"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.not_equal, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.not_equal, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Not_Equal_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_comparison_operations_benchmarks(ctx, inputs):
    """Helper to run all Comparison operator benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.


    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    comparison_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        comparison_operations_results.append(benchmark_ref.get_benchmark_results())

    return comparison_operations_results
