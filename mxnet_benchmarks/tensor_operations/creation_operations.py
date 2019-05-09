import mxnet as mx
import mxnet.ndarray as nd

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import nd_forward_and_time, get_mx_ndarray

""" Performance benchmark tests for MXNet NDArray Creation Operations
1. Zeros
2. Ones
3. Zeros_like
4. Ones_like
5. full
6. arange

TODO:
1. Logging - Info, Error and Debug
2. May be refactor and add MXNetUnaryOperatorBenchmarkBase and move all common code over there?
"""


class Zeros(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Zero filled Tensor Creation operation.

    By default benchmark the creation of a 1024*1024 Zero Tensors with precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"shape": (1024, 1024),
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.zeros, runs=self.warmup, shape=self.inputs["shape"],
                                   ctx=self.ctx, dtype=self.inputs["dtype"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.zeros, runs=self.runs, shape=self.inputs["shape"],
                                          ctx=self.ctx, dtype=self.inputs["dtype"])

        self.results["MX_Zeros_Forward_Time"] = exe_time / self.runs


class Ones(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark One filled Tensor Creation operation.

    By default benchmark the creation of a 1024*1024 Ones Tensors with precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"shape": (1024, 1024),
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.ones, runs=self.warmup, shape=self.inputs["shape"],
                                   ctx=self.ctx, dtype=self.inputs["dtype"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.ones, runs=self.runs, shape=self.inputs["shape"],
                                          ctx=self.ctx, dtype=self.inputs["dtype"])

        self.results["MX_Ones_Forward_Time"] = exe_time / self.runs


class ZerosLike(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Zero filled Tensor Creation operation based on another Tensor.

    By default benchmark the creation of a Zeros Tensor like another 1024*1024 Tensor with precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"data": (1024, 1024),
                              "initializer": nd.normal,
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
        _, _ = nd_forward_and_time(F=nd.zeros_like, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.zeros_like, runs=self.runs, data=self.data)

        self.results["MX_Zeros_Like_Forward_Time"] = exe_time / self.runs


class OnesLike(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark One filled Tensor Creation operation based on another Tensor.

    By default benchmark the creation of a Ones Tensor like another 1024*1024 Tensor with precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"data": (1024, 1024),
                              "initializer": nd.normal,
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
        _, _ = nd_forward_and_time(F=nd.ones_like, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.ones_like, runs=self.runs, data=self.data)

        self.results["MX_Ones_Like_Forward_Time"] = exe_time / self.runs


class Full(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark the creation of a Tensor with given shape and filled with a given constant value.

    By default benchmark the creation of Tensor of shape 1024*1024 filled with value 1.0 and with precision - 'float32'

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"shape": (1024, 1024),
                              "val": 1.0,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.full, runs=self.warmup,
                                   ctx=self.ctx,
                                   shape=self.inputs["shape"],
                                   val=self.inputs["val"],
                                   dtype=self.inputs["dtype"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.full, runs=self.warmup,
                                          ctx=self.ctx,
                                          shape=self.inputs["shape"],
                                          val=self.inputs["val"],
                                          dtype=self.inputs["dtype"])

        self.results["MX_Full_Forward_Time"] = exe_time / self.runs


class Arange(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark the creation of a Tensor in a given range.

    By default benchmark the creation of Tensor with values ranging from 1 to 100 with 10 repetition of each value with
    precision - 'float32'

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default inputs
        default_parameters = {"start": 1,
                              "stop": 100,
                              "step": 1.0,
                              "repeat": 10,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=nd.arange, runs=self.warmup,
                                   ctx=self.ctx,
                                   start=self.inputs["start"],
                                   stop=self.inputs["stop"],
                                   step=self.inputs["step"],
                                   repeat=self.inputs["repeat"],
                                   dtype=self.inputs["dtype"])

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=nd.arange, runs=self.runs,
                                          ctx=self.ctx,
                                          start=self.inputs["start"],
                                          stop=self.inputs["stop"],
                                          step=self.inputs["step"],
                                          repeat=self.inputs["repeat"],
                                          dtype=self.inputs["dtype"])

        self.results["MX_Arange_Forward_Time"] = exe_time / self.runs


# Utilities
def run_all_tensor_creation_operations_benchmarks(ctx, inputs):
    """Helper to run Exponential and Log operator benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.


    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    creation_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        creation_operations_results.append(benchmark_ref.get_benchmark_results())

    return creation_operations_results
