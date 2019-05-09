import mxnet as mx
import mxnet.ndarray as nd

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from utils.profiler_utils import timer

"""
MXNet's Custom Operator Benchmark Tests.

It does a simple element wise addition to make sure computation
is not too much and we can see custom operator logistics overhead.

1. Tests Custom v/s Imperative (Native NDArray)
2. Tests Custom v/s Symbolic (Native Symbol with Simple Bind)
"""


# 1. Define Custom Operator - Element wise Addition Multiplication
class CustomAddOne(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0] + 1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("CustomAddOne")
class CustomAddOneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CustomAddOneProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['in']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        # inputs, outputs, aux
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return CustomAddOne()


# 2. Benchmarks
class CustomOpElementwiseAdd(MXNetOperatorBenchmarkBase):
    """Helps to benchmark MXNet's Custom Op for Elementwise addition on a (1000, 1) tensor.
    Performs both forward and backward operation.

    This test mainly uncovers core custom op overhead in MXNet.

    Benchmark will be done on the following operation:
    native_add -> native_add -> native_add -> CUSTOM_ADD -> native_add -> native_add -> native_add

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (1000, 1),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

    @timer
    def _run_forward_backward_benchmark(self, runs, x):
        for _ in range(runs):
            with mx.autograd.record():
                # Forward
                res1 = x + 1
                res2 = res1 + 1
                res3 = res2 + 1
                res4 = nd.Custom(res3, name="customaddone", op_type="CustomAddOne")
                res5 = res4 + 1
                res6 = res5 + 1
                res7 = res6 + 1

            # Backward
            res7.backward()
            nd.waitall()

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = self._run_forward_backward_benchmark(runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = self._run_forward_backward_benchmark(runs=self.runs, x=self.data)

        self.results["MX_CustomOp_Elementwise_Add_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_customop_operations_benchmarks(ctx, inputs):
    """Helper to run all MXNet custom op benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    customop_operations_results = []

    benchmark_ref = CustomOpElementwiseAdd(ctx=ctx, inputs=inputs)
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
    customop_operations_results.append(benchmark_ref.get_benchmark_results())

    return customop_operations_results
