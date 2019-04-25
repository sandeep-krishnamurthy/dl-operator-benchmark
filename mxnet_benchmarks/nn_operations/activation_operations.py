import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray

""" Performance benchmark Tests for MXNet Gluon Activation Layers.
1. LeakyRelu
2. PRelu
3. Sigmoid
4. Softmax
5. Log_Softmax
6. Activation
"""


class LeakyRelu(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon LeakyRelu Block.

    By default, benchmarks both forward and backward pass on the LeakyRelu block on (32, 3, 256, 256) input with
    'float32' precision. Use alpha=0.01.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        if inputs is None:
            inputs = {"data": (32, 3, 256, 256),
                      "data_initializer": nd.normal,
                      "alpha": 0.01,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.LeakyReLU(alpha=self.inputs["alpha"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_LeakyRelu_Forward_Backward_Time"] = exe_time / self.runs


class PRelu(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon PRelu Block.

    By default, benchmarks both forward and backward pass on the PRelu block on (32, 3, 256, 256) input with
    'float32' precision. Use zero initializer to initialize alpha (Parameter of shape same as input and value
    learnt through training).

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        if inputs is None:
            inputs = {"data": (32, 3, 256, 256),
                      "data_initializer": nd.normal,
                      "alpha_initializer": 'zeros',
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.PReLU(alpha_initializer=self.inputs["alpha_initializer"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_PRelu_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_basic_operations_benchmarks():
    """Helper to run all Gluon Activation Layer benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = LeakyRelu()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = PRelu()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()


run_all_gluon_nn_basic_operations_benchmarks()
