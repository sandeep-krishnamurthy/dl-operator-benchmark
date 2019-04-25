import mxnet as mx
import mxnet.ndarray as nd

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

""" Performance benchmark tests for MXNet NDArray GEMM Operations
1. dot
2. batch_dot

TODO:
1. As part of default tests, following needs to be added:
    1.1 Sparse dot. (csr, default) -> row_sparse
    1.2 Sparse dot. (csr, row_sparse) -> default
    1.3 With Transpose of lhs
    1.4 With Transpose of rhs
2. 1D array: inner product of vectors
3. Logging - Info, Error and Debug
"""


class Dot(MXNetOperatorBenchmarkBase):
    """Helps to Benchmark Tensor Dot operation.

    By default benchmark both forward and backward Dot operation on Tensor of shape (1024, 1024) and Tensor of shape
    (1024, 1000). By default, Tensors are dense and of precision - 'float32'. This defaults are choosen to mimic
    a last dense layer of a 1000 class classification network.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        if inputs is None:
            inputs = {"lhs": (1024, 1024),
                      "rhs": (1024, 1000),
                      "initializer": nd.normal,
                      "transpose_a": False,
                      "transpose_b": False,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

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
        _, _ = nd_forward_backward_and_time(F=nd.dot, runs=self.warmup, lhs=self.lhs, rhs=self.rhs,
                                            transpose_a=self.inputs["transpose_a"],
                                            transpose_b=self.inputs["transpose_b"])

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.dot, runs=self.runs, lhs=self.lhs, rhs=self.rhs,
                                                   transpose_a=self.inputs["transpose_a"],
                                                   transpose_b=self.inputs["transpose_b"])

        self.results["MX_Dot_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gemm_operations_benchmarks():
    """Helper to run all GEMM operators (dot, batch_dot) benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = Dot()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()


run_all_gemm_operations_benchmarks()
