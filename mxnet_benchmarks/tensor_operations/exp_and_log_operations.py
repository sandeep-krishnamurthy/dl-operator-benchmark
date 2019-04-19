import mxnet as mx
import mxnet.ndarray as nd

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

""" Performance benchmark tests for MXNet NDArray Exponential and Logarithms Operations
1. exp
2. log

TODO:
1. Logging - Info, Error and Debug
2. May be refactor and add MXNetUnaryOperatorBenchmarkBase and move all common code over there?
"""


class Log(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Log operation.

    By default benchmark both forward and backward element_wise Log operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (1024, 1024),
                      "initializer": nd.normal,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.log, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.log, runs=self.runs, data=self.data)

        self.results["MX_Log_Forward_Backward_Time"] = exe_time / self.runs


class Exp(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Exponential power operation.

    By default benchmark both forward and backward element_wise Exponential operation on a
    1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (1024, 1024),
                      "initializer": nd.normal,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.exp, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.exp, runs=self.runs, data=self.data)

        self.results["MX_Exp_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_exponential_and_log_operations_benchmarks():
    """Helper to run Exponential and Log operator benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = Log()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = Exp()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
