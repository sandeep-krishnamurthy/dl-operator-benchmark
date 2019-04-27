import argparse
import json

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

"""Driver program to kick off benchmark tasks.

TODO
1. Logging and other useful information.
"""


def _run_all_mxnet_operator_benchmarks():
    """Run all the MXNet operators (NDArray and Gluon) benchmarks with default inputs.

    :return: Dictionary of benchmark results.

    """
    mxnet_operator_benchmark_results = []

    # *************************MXNET TENSOR OPERATOR BENCHMARKS*****************************

    # Run all Tensor creation operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_tensor_creation_operations_benchmarks())

    # Run all Arithmetic operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_arithmetic_operations_benchmarks())

    # Run all Comparison operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_comparison_operations_benchmarks())

    # Run all Logical Comparison operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_logical_comparison_operations_benchmarks())

    # Run all Exp and Log operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_exponential_and_log_operations_benchmarks())

    # Run all GEMM operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gemm_operations_benchmarks())

    # Run all sorting and searching operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_sort_and_search_operations_benchmarks())

    # Run all Powers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_powers_operations_benchmarks())

    # ************************ MXNET GLUON NN LAYERS BENCHMARKS ****************************

    # Run all Gluon NN Basic Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_basic_operations_benchmarks())

    # Run all Gluon NN Activation Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_activation_operations_benchmarks())

    # Run all Gluon Loss Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_loss_operations_benchmarks())

    # Run all Gluon Normalization Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_normalization_operations_benchmarks())

    # Run all Gluon Convolution Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_convolution_operations_benchmarks())

    # Run all Gluon Pooling Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_nn_pooling_operations_benchmarks())

    # Run all Gluon Recurrent Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_all_gluon_recurrent_operations_benchmarks())

    # Run MXNet Custom Op Benchmarks with default input values
    mxnet_operator_benchmark_results.extend(run_customop_operations_benchmarks())

    # ****************************** PREPARE FINAL RESULTS ********************************
    final_benchmark_result_map = merge_map_list(mxnet_operator_benchmark_results)
    return final_benchmark_result_map


# CLI Parser

# 1. GET USER INPUTS
parser = argparse.ArgumentParser(description='Run all the MXNet operators (NDArray and Gluon) benchmarks with default '
                                             'inputs')

# TODO
parser.add_argument('--ctx', type=str, default='cpu', help='Global context to run all benchmarks. By default, cpu on a '
                                                           'CPU machine, gpu(0) on a GPU machine. '
                                                           'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
# TODO
parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
                                                                 'float32. Valid Inputs - float32, float64.')
# TODO - md, csv
parser.add_argument('--output-format', type=str, default='json',
                    help='Benchmark result output format. By default, json. '
                         'Valid Inputs - json, md, csv')

parser.add_argument('--output-file', type=str, default='./mxnet_operator_benchmarks.json', help='Name and path for the '
                                                                                                'output file.')

# TODO - Input validation
user_options = parser.parse_args()
print("Running MXNet operator benchmarks with the following options: ")
print(user_options)

# 2. RUN BENCHMARKS
final_benchmark_result_map = _run_all_mxnet_operator_benchmarks()

# 3. PREPARE OUTPUTS
if user_options.output_format == 'json':
    # Save as JSON
    with open(user_options.output_file, "w") as result_file:
        json.dump(final_benchmark_result_map, result_file, indent=4)
else:
    print("Invalid output file format - {}. Supported - 'json'".format(user_options.output_format))
