import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import loss

from mxnet_benchmarks.MXNetOperatorBenchmark import MXNetOperatorBenchmarkBase
from mxnet_benchmarks.utils.gluon_utils import block_forward_and_time
from mxnet_benchmarks.utils.ndarray_utils import get_mx_ndarray

""" Performance benchmark tests for MXNet Gluon Loss Layers
1. L1Loss
2. L2Loss
3. SigmoidBinaryCrossEntropyLoss
4. SoftmaxCrossEntropyLoss
5. KLDivLoss (TODO)
6. HuberLoss (TODO)
7. HingeLoss (TODO)
8. SquaredHingeLoss (TODO)
9. LogisticLoss (TODO)
10. TripletLoss (TODO)
11. CTCLoss (TODO)
"""


class L1Loss(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon L1Loss Block.

    By default, benchmarks both forward pass on the L1Loss block on pred tensor of shape
    (32, 1000) and Label of shape (1000,1) to mimic batch_size=32 and 1000 class classification.
    By default, use weight=1.0 just to force the weight computation.
    By default, uses 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default prediction is (32, 1000, 1) to mimic a prediction of batch_size=32 and 1000 class classification.
        default_parameters = {"pred": (32, 1000, 1),
                              "pred_initializer": nd.normal,
                              "label": (32, 1000, 1),
                              "label_initializer": nd.normal,
                              "weight": 1.0,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.pred = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["pred"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["pred_initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.label = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["label"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["label_initializer"],
                                    attach_grad=self.inputs["run_backward"])

        self.block = loss.L1Loss(weight=self.inputs["weight"], batch_axis=0)

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_and_time(block=self.block, runs=self.warmup, pred=self.pred, label=self.label)

        # Run Benchmarks
        exe_time, _ = block_forward_and_time(block=self.block, runs=self.runs, pred=self.pred, label=self.label)

        self.results["MX_Gluon_Imperative_L1Loss_Forward_Time"] = exe_time / self.runs


class L2Loss(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon L2Loss Block.

    By default, benchmarks both forward pass on the L2Loss block on pred tensor of shape
    (32, 1000, 1) and Label of shape (32, 1000,1) to mimic batch_size=32 and 1000 class classification.
    By default, use weight=1.0 just to force the weight computation.
    By default, uses 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default prediction is (32, 1000, 1) to mimic a prediction of batch_size=32 and 1000 class classification.
        default_parameters = {"pred": (32, 1000, 1),
                              "pred_initializer": nd.normal,
                              "label": (32, 1000, 1),
                              "label_initializer": nd.normal,
                              "weight": 1.0,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.pred = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["pred"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["pred_initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.label = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["label"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["label_initializer"],
                                    attach_grad=self.inputs["run_backward"])

        self.block = loss.L2Loss(weight=self.inputs["weight"], batch_axis=0)

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_and_time(block=self.block, runs=self.warmup, pred=self.pred, label=self.label)

        # Run Benchmarks
        exe_time, _ = block_forward_and_time(block=self.block, runs=self.runs, pred=self.pred, label=self.label)

        self.results["MX_Gluon_Imperative_L2Loss_Forward_Time"] = exe_time / self.runs


class SigmoidBinaryCrossEntropyLoss(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon SigmoidBinaryCrossEntropyLoss Block.

    By default, benchmarks both forward pass on the SigmoidBinaryCrossEntropyLoss block on pred tensor of shape
    (32, 1000, 1) and Label of shape (32, 1000,1) to mimic batch_size=32 and 1000 class classification.
    By default, use weight=1.0 just to force the weight computation.
    By default, uses 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default prediction is (32, 1000, 1) to mimic a prediction of batch_size=32 and 1000 class classification.
        default_parameters = {"pred": (32, 1000, 1),
                              "pred_initializer": nd.ones,
                              "label": (32, 1000, 1),
                              "label_initializer": nd.zeros,
                              "weight": 1.0,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.pred = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["pred"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["pred_initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.label = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["label"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["label_initializer"],
                                    attach_grad=self.inputs["run_backward"])

        self.block = loss.SigmoidBCELoss(from_sigmoid=False, weight=self.inputs["weight"], batch_axis=0)

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_and_time(block=self.block, runs=self.warmup, pred=self.pred, label=self.label)

        # Run Benchmarks
        exe_time, _ = block_forward_and_time(block=self.block, runs=self.runs, pred=self.pred, label=self.label)

        self.results["MX_Gluon_Imperative_SigmoidBinaryCrossEntropyLoss_Forward_Time"] = exe_time / self.runs


class SoftmaxCrossEntropyLoss(MXNetOperatorBenchmarkBase):
    """Helps to benchmark Gluon SoftmaxCrossEntropyLoss Block.

    By default, benchmarks both forward pass on the SoftmaxCrossEntropyLoss block on pred tensor of shape
    (32, 1000, 1) and Label of shape (32, 1000,1) to mimic batch_size=32 and 1000 class classification.
    By default, use weight=1.0 just to force the weight computation.
    By default, uses 'float32' precision.

    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=50, inputs=None):
        # Set the default Inputs.
        # Default prediction is (32, 1000, 1) to mimic a prediction of batch_size=32 and 1000 class classification.
        default_parameters = {"pred": (32, 1000, 1),
                              "pred_initializer": nd.normal,
                              "label": (32, 1000, 1),
                              "label_initializer": nd.normal,
                              "weight": 1.0,
                              "axis": -1,
                              "sparse_label": False,
                              "run_backward": False,
                              "dtype": "float32"}

        super().__init__(ctx=ctx, warmup=warmup, runs=runs, default_parameters=default_parameters,
                         custom_parameters=inputs)

        # Create a random prediction and label tensor
        self.pred = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["pred"],
                                   dtype=self.inputs["dtype"],
                                   initializer=self.inputs["pred_initializer"],
                                   attach_grad=self.inputs["run_backward"])
        self.label = get_mx_ndarray(ctx=self.ctx, in_tensor=self.inputs["label"],
                                    dtype=self.inputs["dtype"],
                                    initializer=self.inputs["label_initializer"],
                                    attach_grad=self.inputs["run_backward"])

        self.block = loss.SoftmaxCrossEntropyLoss(axis=self.inputs["axis"],
                                                  sparse_label=self.inputs["sparse_label"],
                                                  from_logits=False,
                                                  weight=self.inputs["weight"],
                                                  batch_axis=0)

        self.block.initialize(ctx=self.ctx)

    def run_benchmark(self):
        # Warm up, ignore execution time value
        _, _ = block_forward_and_time(block=self.block, runs=self.warmup, pred=self.pred, label=self.label)

        # Run Benchmarks
        exe_time, _ = block_forward_and_time(block=self.block, runs=self.runs, pred=self.pred, label=self.label)

        self.results["MX_Gluon_Imperative_SoftmaxCrossEntropyLoss_Forward_Time"] = exe_time / self.runs


# Utilities
def run_all_gluon_nn_loss_operations_benchmarks():
    """Helper to run all Gluon Loss Layer benchmarks. Just runs the benchmarks with default input values.
    This just a utility to run benchmarks with all default input values.

    TODO: Capture results in a clean dictionary rather than printing everything to console.
    """
    nn_loss_operations_results = []

    benchmark_ref = L1Loss()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
    nn_loss_operations_results.append(benchmark_ref.get_benchmark_results())

    benchmark_ref = L2Loss()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
    nn_loss_operations_results.append(benchmark_ref.get_benchmark_results())

    benchmark_ref = SigmoidBinaryCrossEntropyLoss()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
    nn_loss_operations_results.append(benchmark_ref.get_benchmark_results())

    benchmark_ref = SoftmaxCrossEntropyLoss()
    benchmark_ref.run_benchmark()
    benchmark_ref.print_benchmark_results()
    nn_loss_operations_results.append(benchmark_ref.get_benchmark_results())

    return nn_loss_operations_results
