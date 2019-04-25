import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time

""" Performance benchmark tests for MXNet Gluon Basic NN Layers
1. Dense
2. Lambda
3. Flatten
4. Embedding

TODO:
1. Logging - Info, Error and Debug
3. Probably we can refactor the common logic of all these binary operations into a parent
   MXNetGluonBlockBenchmarkBase?
"""


class Dense(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Dense Block.

    By default, benchmarks both forward and backward pass on the Dense block with 256 units, no activation and
    an input data of size (512, 512). By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (1, 1024) to mimic an input of batch_size=1 and a sample image of size 512*512.
        # Default number of units 256 is referred from ResNet architecture as commonly used Dense Layer size.
        # Default activation is None because we want to benchmark just dense layer operation.
        if inputs is None:
            inputs = {"data": (512, 512),
                      "units": 256,
                      "activation": None,
                      "use_bias": True,
                      "flatten": True,
                      "data_initializer": nd.normal,
                      "weight_initializer": "Xavier",
                      "bias_initializer": "Zeros",
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Dense(units=self.inputs["units"],
                              activation=self.inputs["activation"],
                              use_bias=self.inputs["use_bias"],
                              flatten=self.inputs["flatten"],
                              dtype=self.inputs["dtype"],
                              weight_initializer=self.inputs["weight_initializer"],
                              bias_initializer=self.inputs["bias_initializer"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Dense_Forward_Backward_Time"] = exe_time / self.runs


class Flatten(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Flatten Block.

    By default, benchmarks both forward and backward pass on the Flatten block on (128, 512, 512) input with
    'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (128, 512, 512) to mimic an input of batch_size=128 and a sample image of size 512*512.
        if inputs is None:
            inputs = {"data": (128, 512, 512),
                      "data_initializer": nd.normal,
                      "run_backward": True,
                      "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, inputs=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Flatten()

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Flatten_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_basic_operations_benchmarks():
    """Helper to run all Gluon Basic NN Layer benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    benchmark_ref = Dense()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()

    benchmark_ref = Flatten()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()


run_all_gluon_nn_basic_operations_benchmarks()
