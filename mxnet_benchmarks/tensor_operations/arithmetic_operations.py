import mxnet as mx
import mxnet.ndarray as nd

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_and_time, nd_forward_backward_and_time

"""Performance benchmark tests for MXNet NDArray Arithmetic Operations
1. Add
2. Sub
3. Mul
4. Div
5. Mod
6. Pow
7. Neg
8. iadd (In place Add with +=)
9. isub (In place Sub with -=)
10. imul (In place Mul with *=)
11. idiv (In place Div with /=)
12. imod (In place Mod with %=)

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
"""


class Add(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Add operation.

    By default benchmark both forward and backward element_wise tensor addition
    of 1024*1024 tensor of precision - 'float32'.

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
        _, _ = nd_forward_backward_and_time(F=nd.add, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.add, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Add_Forward_Backward_Time"] = exe_time / self.runs


class Subtract(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Subtract operation.

    By default benchmark both forward and backward element_wise tensor subtraction
    of 1024*1024 tensor of precision - 'float32'.

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
        _, _ = nd_forward_backward_and_time(F=nd.subtract, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.subtract, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Subtract_Forward_Backward_Time"] = exe_time / self.runs


class Multiply(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Multiply operation.

    By default benchmark both forward and backward element_wise tensor multiply
    of 1024*1024 tensor of precision - 'float32'.

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
        _, _ = nd_forward_backward_and_time(F=nd.multiply, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.multiply, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Multiply_Forward_Backward_Time"] = exe_time / self.runs


class Divide(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Divide operation.

    By default benchmark both forward and backward element_wise tensor subtraction
    of 1024*1024 tensor of precision - 'float32'.

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
        _, _ = nd_forward_backward_and_time(F=nd.divide, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.divide, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Divide_Forward_Backward_Time"] = exe_time / self.runs


class Modulo(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Modulo operation.

    By default benchmark both forward and backward element_wise tensor modulo
    of 1024*1024 tensor of precision - 'float32'.

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
        _, _ = nd_forward_backward_and_time(F=nd.modulo, runs=self.warmup, lhs=self.lhs, rhs=self.rhs)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.modulo, runs=self.runs, lhs=self.lhs, rhs=self.rhs)

        self.results["MX_Modulo_Forward_Backward_Time"] = exe_time / self.runs


class Power(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Power operation.

    By default benchmark both forward and backward with 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"base": (1024, 1024),
                              "exp": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.base = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["base"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.exp = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["exp"],
                                  dtype=self.inputs["dtype"],
                                  initializer=self.inputs["initializer"],
                                  attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.power, runs=self.warmup, base=self.base, exp=self.exp)
        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.power, runs=self.runs, base=self.base, exp=self.exp)

        self.results["MX_Power_Forward_Backward_Time"] = exe_time / self.runs


class Negative(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Negative operation.

    By default benchmark both forward and backward with 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_backward_and_time(F=nd.negative, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.negative, runs=self.runs, data=self.data)

        self.results["MX_Negative_Forward_Backward_Time"] = exe_time / self.runs


class IAdd(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Inplace Add operation.

    By default benchmark both forward and backward element_wise tensor Inplace addition
    of 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"this": (1024, 1024),
                              "other": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.this = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["this"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.other = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["other"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["initializer"],
                                    attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=self.this.__iadd__, runs=self.warmup, other=self.other)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=self.this.__iadd__, runs=self.runs, other=self.other)

        self.results["MX_Inplace_Add_Forward_Time"] = exe_time / self.runs


class ISub(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Inplace Subtract operation.

    By default benchmark both forward and backward element_wise tensor Inplace subtract
    of 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"this": (1024, 1024),
                              "other": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)
        self.this = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["this"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.other = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["other"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["initializer"],
                                    attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=self.this.__isub__, runs=self.warmup, other=self.other)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=self.this.__isub__, runs=self.runs, other=self.other)

        self.results["MX_Inplace_Subtract_Forward_Time"] = exe_time / self.runs


class IMul(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Inplace Multiplication operation.

    By default benchmark both forward and backward element_wise tensor Inplace multiplication
    of 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"this": (1024, 1024),
                              "other": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.this = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["this"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.other = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["other"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["initializer"],
                                    attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=self.this.__imul__, runs=self.warmup, other=self.other)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=self.this.__imul__, runs=self.runs, other=self.other)

        self.results["MX_Inplace_Multiply_Forward_Time"] = exe_time / self.runs


class IDiv(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Inplace Division operation.

    By default benchmark both forward and backward element_wise tensor Inplace division
    of 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"this": (1024, 1024),
                              "other": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.this = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["this"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.other = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["other"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["initializer"],
                                    attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=self.this.__idiv__, runs=self.warmup, other=self.other)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=self.this.__idiv__, runs=self.runs, other=self.other)

        self.results["MX_Inplace_Division_Forward_Time"] = exe_time / self.runs


class IMod(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Inplace Modulo operation.

    By default benchmark both forward and backward element_wise tensor Inplace modulo
    of 1024*1024 tensor of precision - 'float32'.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"this": (1024, 1024),
                              "other": (1024, 1024),
                              "initializer": nd.normal,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.this = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["this"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.other = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["other"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["initializer"],
                                    attach_grad=self.inputs["run_backward"])

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = nd_forward_and_time(F=self.this.__imod__, runs=self.warmup, other=self.other)

        # Run Benchmarks
        exe_time, _ = nd_forward_and_time(F=self.this.__imod__, runs=self.runs, other=self.other)

        self.results["MX_Inplace_Modulo_Forward_Time"] = exe_time / self.runs


# Utilities
def run_all_arithmetic_operations_benchmarks():
    """Helper to run all Arithmetic operator benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    arithmetic_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls()
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        arithmetic_operations_results.append(benchmark_ref.get_benchmark_results())

    return arithmetic_operations_results
