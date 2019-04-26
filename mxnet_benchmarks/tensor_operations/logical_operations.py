import mxnet as mx
from mxnet import nd

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

"""Performance benchmark tests for MXNet NDArray Logical Operations
1. logical_and
2. logical_or
3. logical_xor
4. logical_not

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
3. Probably we can refactor the common logic of all these binary operations into a parent
   MXNetBinaryOperatorBenchmarkBase?
"""


class LogicalAnd(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor LogicalAnd operation.

    By default benchmark both forward and backward element_wise LogicalAnd operation on a
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
        _, _ = nd_forward_backward_and_time(F=nd.logical_and, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.logical_and, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Logical_And_Forward_Backward_Time"] = exe_time / self.runs


class LogicalOr(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor LogicalOr operation.

    By default benchmark both forward and backward element_wise LogicalOr operation on a
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
        _, _ = nd_forward_backward_and_time(F=nd.logical_or, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.logical_or, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Logical_Or_Forward_Backward_Time"] = exe_time / self.runs


class LogicalXor(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor LogicalXor operation.

    By default benchmark both forward and backward element_wise LogicalXor operation on a
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
        _, _ = nd_forward_backward_and_time(F=nd.logical_xor, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.logical_xor, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Logical_Xor_Forward_Backward_Time"] = exe_time / self.runs


class LogicalNot(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor LogicalNot operation.

    By default benchmark both forward and backward element_wise LogicalNot operation on a
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
        _, _ = nd_forward_backward_and_time(F=nd.logical_not, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.logical_not, runs=self.runs, data=self.data)

        self.results["MX_Logical_Not_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_logical_comparison_operations_benchmarks():
    """Helper to run all Logical Comparison operator benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = LogicalAnd()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = LogicalOr()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = LogicalXor()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = LogicalNot()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
