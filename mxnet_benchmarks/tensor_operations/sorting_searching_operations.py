import mxnet as mx
import mxnet.ndarray as nd

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_and_time

""" Performance benchmark tests for MXNet NDArray Sorting and Searching Operations
1. sort
2. argsort
3. topk
4. argmax
5. argmin

TODO:
1. Sort and Argsort
    1.1 Descending Order
    1.2 Flatten and sort
2. TopK
    1.1 K being a very small number (ex: 1) on a axis with 1000 values.
3. argmax_channel (This is same as argmax with axis=-1
"""


class Sort(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Sort operation.

    By default, benchmark forward Sort operation on a 1024*1000 tensor on last axis in ascending
    order.
    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1000),
                              "initializer": nd.normal,
                              "axis": -1,
                              "is_ascend": True,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.sort, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                   is_ascend=self.inputs["is_ascend"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.sort, runs=self.runs, data=self.data, axis=self.inputs["axis"],
                                          is_ascend=self.inputs["is_ascend"])

        self.results["MX_Sort_Forward_Time"] = exe_time / self.runs


class ArgSort(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor ArgSort operation.

    By default, benchmark forward ArgSort operation on a 1024*1000 tensor on last axis in ascending
    order.
    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1000),
                              "initializer": nd.normal,
                              "axis": -1,
                              "is_ascend": True,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.argsort, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                   is_ascend=self.inputs["is_ascend"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.argsort, runs=self.runs, data=self.data,
                                          axis=self.inputs["axis"],
                                          is_ascend=self.inputs["is_ascend"])

        self.results["MX_ArgSort_Forward_Time"] = exe_time / self.runs


class TopK(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor TopK operation.

    By default, benchmark forward TopK operation with K=10, on a 1024*1000 tensor on last axis,
    in ascending order. Result being the index of element.
    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1000),
                              "initializer": nd.normal,
                              "axis": -1,
                              "k": 10,
                              "ret_typ": "indices",
                              "is_ascend": True,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.topk, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                   k=self.inputs["k"], ret_typ=self.inputs["ret_typ"],
                                   is_ascend=self.inputs["is_ascend"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.topk, runs=self.runs, data=self.data, axis=self.inputs["axis"],
                                          k=self.inputs["k"], ret_typ=self.inputs["ret_typ"],
                                          is_ascend=self.inputs["is_ascend"])

        self.results["MX_TopK_Forward_Backward_Time"] = exe_time / self.runs


class ArgMax(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor ArgMax operation.

    By default, benchmark forward ArgMax operation on a 1024*1000 tensor on last axis.
    Result being the index of max element in the last axis.

    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1000),
                              "initializer": nd.normal,
                              "axis": -1,
                              "keepdims": False,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.argmax, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                   keepdims=self.inputs["keepdims"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.argmax, runs=self.runs, data=self.data, axis=self.inputs["axis"],
                                          keepdims=self.inputs["keepdims"])

        self.results["MX_ArgMax_Forward_Backward_Time"] = exe_time / self.runs


class ArgMin(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor ArgMin operation.

    By default, benchmark forward ArgMin operation on a 1024*1000 tensor on last axis.
    Result being the index of min element in the last axis.

    By default, uses precision - 'float32' for tensors.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1000),
                              "initializer": nd.normal,
                              "axis": -1,
                              "keepdims": False,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.argmin, runs=self.warmup, data=self.data, axis=self.inputs["axis"],
                                   keepdims=self.inputs["keepdims"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.argmin, runs=self.runs, data=self.data, axis=self.inputs["axis"],
                                          keepdims=self.inputs["keepdims"])

        self.results["MX_ArgMin_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_sort_and_search_operations_benchmarks():
    """Helper to run all Sort and Search operator benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    sort_search_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls()
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        sort_search_operations_results.append(benchmark_ref.get_benchmark_results())

    return sort_search_operations_results
