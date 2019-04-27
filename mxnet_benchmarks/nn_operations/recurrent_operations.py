import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import rnn

from utils.common_utils import get_class_members_in_module
from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray
from mxnet_benchmarks.utils.gluon_utils import block_forward_backward_and_time

""" Performance benchmark tests for MXNet Gluon Recurrent Layers
1. RNN
2. LSTM
3. GRU

TODO:
4. RNNCell
5. LSTMCell
6. GRUCell
7. RecurrentCell
8. SequentialRNNCell
9. BidirectionalCell
"""


class RNN(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon RNN Block.

    By default, benchmarks both forward and backward pass on the RNN block with 100 hidden units, 1 layer,
    relu activation, input in TNC layout and (25, 32, 256) shape, no dropout, single directional RNN operation.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (25, 32, 256),
                              "data_initializer": nd.normal,
                              "hidden_size": 100,
                              "num_layers": 1,
                              "activation": "relu",
                              "layout": "TNC",
                              "dropout": 0,
                              "bidirectional": False,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = rnn.RNN(hidden_size=self.inputs["hidden_size"],
                             num_layers=self.inputs["num_layers"],
                             activation=self.inputs["activation"],
                             layout=self.inputs["layout"],
                             dropout=self.inputs["dropout"],
                             bidirectional=self.inputs["bidirectional"],
                             dtype=self.inputs["dtype"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_RNN_Forward_Backward_Time"] = exe_time / self.runs


class LSTM(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon LSTM Block.

    By default, benchmarks both forward and backward pass on the LSTM block with 100 hidden units, 1 layer,
    input in TNC layout and (25, 32, 256) shape, no dropout, single directional LSTM operation.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (25, 32, 256),
                              "data_initializer": nd.normal,
                              "hidden_size": 100,
                              "num_layers": 1,
                              "layout": "TNC",
                              "dropout": 0,
                              "bidirectional": False,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = rnn.LSTM(hidden_size=self.inputs["hidden_size"],
                              num_layers=self.inputs["num_layers"],
                              layout=self.inputs["layout"],
                              dropout=self.inputs["dropout"],
                              bidirectional=self.inputs["bidirectional"],
                              dtype=self.inputs["dtype"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_LSTM_Forward_Backward_Time"] = exe_time / self.runs


class GRU(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon GRU Block.

    By default, benchmarks both forward and backward pass on the GRU block with 100 hidden units, 1 layer,
    input in TNC layout and (25, 32, 256) shape, no dropout, single directional GRU operation.

    By default run on 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs
        default_parameters = {"data": (25, 32, 256),
                              "data_initializer": nd.normal,
                              "hidden_size": 100,
                              "num_layers": 1,
                              "layout": "TNC",
                              "dropout": 0,
                              "bidirectional": False,
                              "run_backward": True,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        self.data = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["data"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["data_initializer"],
                                   attach_grad=self.inputs["run_backward"])

        self.block = rnn.GRU(hidden_size=self.inputs["hidden_size"],
                             num_layers=self.inputs["num_layers"],
                             layout=self.inputs["layout"],
                             dropout=self.inputs["dropout"],
                             bidirectional=self.inputs["bidirectional"],
                             dtype=self.inputs["dtype"])

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_backward_and_time(block=self.block, runs=self.warmup, x=self.data)

        # Run Benchmarks
        exe_time, _ = block_forward_backward_and_time(block=self.block, runs=self.runs, x=self.data)

        self.results["MX_Gluon_Imperative_GRU_Forward_Backward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_recurrent_operations_benchmarks():
    """Helper to run all Gluon Recurrent Layers benchmarks. Just runs the benchmarks with default input values.
    This is just a utility to run benchmarks with all default input values.

    :return: list[dict], list of dictionary of benchmark results. Each item in the list is a dictionary of benchmark
                         results per operator.

    """
    recurrent_operations_results = []

    members = get_class_members_in_module(__name__)

    for _, cls in members:
        benchmark_ref = cls()
        benchmark_ref.run_benchmark()
        benchmark_ref.print_benchmark_results()
        recurrent_operations_results.append(benchmark_ref.get_benchmark_results())

    return recurrent_operations_results
