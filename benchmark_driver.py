import json

from utils.common_utils import merge_map_list

from mxnet_benchmarks.tensor_operations.creation_operations import run_all_tensor_creation_operations_benchmarks
from mxnet_benchmarks.tensor_operations.arithmetic_operations import run_all_arithmetic_operations_benchmarks
from mxnet_benchmarks.tensor_operations.comparison_operations import run_all_comparison_operations_benchmarks
from mxnet_benchmarks.tensor_operations.logical_operations import run_all_logical_comparison_operations_benchmarks
from mxnet_benchmarks.tensor_operations.exp_and_log_operations import run_all_exponential_and_log_operations_benchmarks
from mxnet_benchmarks.tensor_operations.gemm_operations import run_all_gemm_operations_benchmarks
from mxnet_benchmarks.tensor_operations.sorting_searching_operations import \
    run_all_sort_and_search_operations_benchmarks

from mxnet_benchmarks.nn_operations.basic_operations import run_all_gluon_nn_basic_operations_benchmarks
from mxnet_benchmarks.nn_operations.activation_operations import run_all_gluon_nn_activation_operations_benchmarks
from mxnet_benchmarks.nn_operations.loss_operations import run_all_gluon_nn_loss_operations_benchmarks
from mxnet_benchmarks.nn_operations.normalization_operations import run_all_gluon_normalization_operations_benchmarks
from mxnet_benchmarks.nn_operations.convolution_operations import run_all_gluon_nn_convolution_operations_benchmarks
from mxnet_benchmarks.nn_operations.pooling_operations import run_all_gluon_nn_pooling_operations_benchmarks
from mxnet_benchmarks.nn_operations.recurrent_operations import run_all_gluon_recurrent_operations_benchmarks

from mxnet_benchmarks.custom_operations.custom_operations import run_customop_operations_benchmarks

"""Driver program to kick off benchmark tasks.

TODO
1. Capture results in proper dictionary and create a comprehensive report.
2. Run benchmarks for various input type and size combination. Below for illustration purpose we use default benchmarks.
3. Logging and other useful information.
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

# TODO: Get the result file and format as input. Ex: result can be markdown or json string rather file etc.
# Save as JSON
with open("mxnet_operator_benchmarks.json", "w") as result_file:
    json.dump(final_benchmark_result_map, result_file, indent=4)
