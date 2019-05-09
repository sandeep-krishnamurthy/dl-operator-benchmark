import mxnet as mx

from utils.common_utils import merge_map_list

from mxnet_benchmarks.custom_operations.custom_operations import run_customop_operations_benchmarks

from mxnet_benchmarks.nd import run_all_tensor_creation_operations_benchmarks, \
    run_all_arithmetic_operations_benchmarks, run_all_comparison_operations_benchmarks, \
    run_all_logical_comparison_operations_benchmarks, run_all_exponential_and_log_operations_benchmarks, \
    run_all_gemm_operations_benchmarks, run_all_sort_and_search_operations_benchmarks, \
    run_all_powers_operations_benchmarks

from mxnet_benchmarks.gluon.nn import run_all_gluon_nn_basic_operations_benchmarks, \
    run_all_gluon_nn_activation_operations_benchmarks, run_all_gluon_nn_pooling_operations_benchmarks, \
    run_all_gluon_normalization_operations_benchmarks, run_all_gluon_nn_convolution_operations_benchmarks

from mxnet_benchmarks.gluon.rnn import run_all_gluon_recurrent_operations_benchmarks

from mxnet_benchmarks.gluon.loss import run_all_gluon_nn_loss_operations_benchmarks

"""Driver program to kick off all MXNet operator benchmark tasks.

TODO
1. Logging and other useful information.
"""


def run_all_mxnet_operator_benchmarks(ctx=mx.cpu(), inputs={}):
    """Run all the MXNet operators (NDArray and Gluon) benchmarks with default inputs.

    :return: Dictionary of benchmark results.

    """
    mxnet_operator_benchmark_results = []

    # *************************MXNET TENSOR OPERATOR BENCHMARKS*****************************

    # Run all Tensor creation operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_tensor_creation_operations_benchmarks(ctx, inputs))

    # Run all Arithmetic operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_arithmetic_operations_benchmarks(ctx, inputs))

    # Run all Comparison operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_comparison_operations_benchmarks(ctx, inputs))

    # Run all Logical Comparison operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_logical_comparison_operations_benchmarks(ctx, inputs))

    # Run all Exp and Log operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_exponential_and_log_operations_benchmarks(ctx, inputs))

    # Run all GEMM operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gemm_operations_benchmarks(ctx, inputs))

    # Run all sorting and searching operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_sort_and_search_operations_benchmarks(ctx, inputs))

    # Run all Powers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_powers_operations_benchmarks(ctx, inputs))

    # ************************ MXNET GLUON NN LAYERS BENCHMARKS ****************************

    # Run all Gluon NN Basic Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_basic_operations_benchmarks(ctx, inputs))

    # Run all Gluon NN Activation Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_activation_operations_benchmarks(ctx, inputs))

    # Run all Gluon Loss Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_loss_operations_benchmarks(ctx, inputs))

    # Run all Gluon Normalization Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_normalization_operations_benchmarks(ctx, inputs))

    # Run all Gluon Convolution Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_convolution_operations_benchmarks(ctx, inputs))

    # Run all Gluon Pooling Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_pooling_operations_benchmarks(ctx, inputs))

    # Run all Gluon Recurrent Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_recurrent_operations_benchmarks(ctx, inputs))

    # Run MXNet Custom Op Benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_customop_operations_benchmarks(ctx, inputs))

    # ****************************** PREPARE FINAL RESULTS ********************************
    final_benchmark_result_map = merge_map_list(mxnet_operator_benchmark_results)
    return final_benchmark_result_map
