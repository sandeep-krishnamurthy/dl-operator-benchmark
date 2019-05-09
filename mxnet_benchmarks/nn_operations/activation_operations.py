import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray, nd_forward_backward_and_time

""" Performance benchmark Tests for MXNet Gluon Activation Layers.
1. LeakyRelu
2. PRelu
3. Activation (Sigmoid)
4. Activation (Softmax) (TODO - GLUON does not have Softmax block, using NDArray APIs for now)
5. Activation (Log_Softmax) (TODO - GLUON does not have Log_Softmax block, using NDArray APIs for now)
6. Activation (tanh)
7. Elu
8. Selu
9. Swish
"""


class LeakyRelu(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon LeakyRelu Activation Block.

    By default, benchmarks both forward and backward pass on the LeakyRelu block on (32, 3, 256, 256) input with
    'float32' precision. Use alpha=0.01.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "alpha": 0.01,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

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
    """Helps to benchmark Gluon PRelu Activation Block.

    By default, benchmarks both forward and backward pass on the PRelu block on (32, 3, 256, 256) input with
    'float32' precision. Use zero initializer to initialize alpha (Parameter of shape same as input and value
    learnt through training).

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "alpha_initializer": 'zeros',
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

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


class Sigmoid(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Sigmoid Activation Block.

    By default, benchmarks both forward and backward pass on the Sigmoid Activation block on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Activation(activation="sigmoid")

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Sigmoid_Forward_Backward_Time"] = exe_time / self.runs


class Softmax(MXNetOperatorBenchmarkBase):
    """Helps to benchmark NDArray Softmax Activation (GLUON doesnot have a softmax block!).

    By default, benchmarks both forward and backward pass on the NDArray Softmax operation on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
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
        _, _ = nd_forward_backward_and_time(F=nd.softmax, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.softmax, runs=self.runs, data=self.data)

        self.results["MX_Softmax_Forward_Backward_Time"] = exe_time / self.runs


class LogSoftmax(MXNetOperatorBenchmarkBase):
    """Helps to benchmark NDArray Log_Softmax Activation (GLUON does not have a log_softmax block!).

    By default, benchmarks both forward and backward pass on the Log_Softmax Activation operation on (32, 3, 256, 256)
    input with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
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
        _, _ = nd_forward_backward_and_time(F=nd.log_softmax, runs=self.warmup, data=self.data)

        # Run Benchmarks
        exe_time, _ = nd_forward_backward_and_time(F=nd.log_softmax, runs=self.runs, data=self.data)

        self.results["MX_Log_Softmax_Forward_Backward_Time"] = exe_time / self.runs


class Tanh(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Tanh Activation Block.

    By default, benchmarks both forward and backward pass on the Tanh Activation block on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Activation(activation="tanh")

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Tanh_Forward_Backward_Time"] = exe_time / self.runs


class Elu(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Elu Activation Block.

    By default, benchmarks both forward and backward pass on the Elu Activation block on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.ELU()

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Elu_Forward_Backward_Time"] = exe_time / self.runs


class Selu(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Selu Activation Block.

    By default, benchmarks both forward and backward pass on the Selu Activation block on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.SELU()

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Selu_Forward_Backward_Time"] = exe_time / self.runs


class Swish(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon Swish Activation Block.

    By default, benchmarks both forward and backward pass on the Swish Activation block on (32, 3, 256, 256) input
    with 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default data is (32, 3, 256, 256) to mimic an input of batch_size=128 and a sample image of size 3*256*256.
        default_parameters = {"data": (32, 3, 256, 256),
                              "data_initializer": nd.normal,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = nn.Swish()

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_Swish_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_activation_operations_benchmarks(ctx, inputs):
    """Helper to run all Gluon Activation Layer benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    activation_operations_results = []
    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls(ctx=ctx, inputs=inputs)
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        activation_operations_results.append(benchmark_ref.get_benchmark_results())

    return activation_operations_results
