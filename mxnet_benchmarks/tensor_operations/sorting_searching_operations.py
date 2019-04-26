import mxnet as mx
import mxnet.ndarray as nd

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

""" Performance benchmark tests for MXNet NDArray Sorting and Searching Operations
1. sort
2. argsort
3. topk
4. argmax
5. argmin
6. argmax_channel

TODO:
1. Sort and Argsort
    1.1 Descending Order
    1.2 Flatten and sort
2. 
"""


class Sort(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Sort operation.

    By default, benchmark both forward and backward Sort operation on a 1024*1000 tensor on last axis in ascending
    order.
    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (1024, 1000),
                      "initializer": nd.normal,
                      "axis": -1,
                      "is_ascend": True,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.sort, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                            is_ascend=self.inputs["is_ascend"])

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.sort, runs=self.runs, data=self.data, axis=self.inputs["axis"],
                                                   is_ascend=self.inputs["is_ascend"])

        self.results["MX_Sort_Forward_Backward_Time"] = exe_time / self.runs


class ArgSort(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor ArgSort operation.

    By default, benchmark both forward and backward ArgSort operation on a 1024*1000 tensor on last axis in ascending
    order.
    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"data": (1024, 1000),
                      "initializer": nd.normal,
                      "axis": -1,
                      "is_ascend": True,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.argsort, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                            is_ascend=self.inputs["is_ascend"])

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.argsort, runs=self.runs, data=self.data,
                                                   axis=self.inputs["axis"],
                                                   is_ascend=self.inputs["is_ascend"])

        self.results["MX_ArgSort_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_sort_and_search_operations_benchmarks():
    """Helper to run all Sort and Search operator benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = Sort()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = ArgSort()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()


run_all_sort_and_search_operations_benchmarks()
