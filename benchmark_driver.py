from mxnet_benchmarks.tensor_operations.arithmetic_operations import run_all_arithmetic_operations_benchmarks
from mxnet_benchmarks.tensor_operations.comparison_operations import run_all_comparison_operations_benchmarks
from mxnet_benchmarks.tensor_operations.exp_and_log_operations import run_all_exponential_and_log_operations_benchmarks

"""Driver program to kick off benchmark tasks.

TODO
1. Capture results in proper dictionary and create a comprehensive report.
2. Run benchmarks for various input type and size combination. Below for illustration purpose we use default benchmarks.
3. Logging and other useful information.
"""

# Run all Arithmetic operations benchmarks with default input values
run_all_arithmetic_operations_benchmarks()

# Run all Comparison operations benchmarks with default input values
run_all_comparison_operations_benchmarks()

# Run all Exp and Log operations benchmarks with default input values
run_all_exponential_and_log_operations_benchmarks()
