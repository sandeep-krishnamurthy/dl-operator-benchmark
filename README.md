# DL Operator Benchmarks
A Python framework for benchmarking operators in [Apache MXNet Deep Learning Library](http://mxnet.incubator.apache.org/).

## Features

1. Individual operator benchmarks to capture "Speed" - operator execution time for `Forward`, `Backward` or `Both Forward and Backward` operations.
2. Individual operator benchmarks to capture "Memory" usage. (TODO - Coming Soon...)
3. Benchmarks for commonly fused operators. Ex: Conv + Relu, Conv + BatchNorm. (TODO - Coming Soon...)
4. Benchmarks for operators with varying inputs to uncover any performance issues due to skewed input data. Example: Measuring operator performance on small input tensors, large input tensors along with average normally used tensor sizes.
5. Support running all or subset of operator benchmarks.
6. Support running operator benchmarks with reasonable default inputs or customize the inputs to the operators. Example: By default, use (1024, 1024) Float32 tensor for Add operator or allow users to specify dtype to be Float64, tensors of shape (10, 100).
7. Support exporting benchmarks results in different formats - stdout (console output), dictionary output, write to JSON, Markdown or CSV files.


Currently supports benchmarking [`NDArray`](http://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html) operators and [`Gluon`](http://mxnet.incubator.apache.org/api/python/gluon/gluon.html) blocks (layers) in MXNet.

## Motivation

Benchmarks are usually done end-to-end for a given Network Architecture. For example: ResNet-50 benchmarks on ImageNet data. This is good measurement of overall performance and health of a deep learning framework. However, it is important to note the following important factors:
1. Users use a lot more operators that are not part of a standard network like ResNet. Example: Tensor manipulation operators like mean, max, topk, argmax, sort etc.   
2. A standard Network Architecture like ResNet-50 is made up of many operators Ex: Convolution2D, Softmax, Dense and more. Consider the following scenarios:
    1. We improved the performance of Convolution2D operator, but due to a bug, Softmax performance went down. Overall, we may observe end to end benchmarks are running fine, we may miss out the performance degradation of a single operator which can accumulate and become untraceable.
    2. You need to see in a given network, which operator is taking maximum time and plan optimization work. With end to end benchmarks, it is hard to get more fine grained numbers at operator level.
3. We need to know on different hardware infrastructure (Ex: CPU with MKLDNN, GPU with NVIDIA CUDA and cuDNN) how different operators performs. With these details, we can plan the optimization work at operator level, which could exponentially boost up end to end performance.
4. You want to have nightly performance tests across all operators in a deep learning framework to catch regressions early. 
5. We can integrate this framework with a CI/CD system to run per operator performance tests for PRs. Example: When a PR modifies the kernel of TransposeConv2D, we can run benchmarks of TransposeConv2D operator to verify performance.

Hence, in this framework, we will build the functionality to allow users and developers of deep learning frameworks to easily run benchmarks for individual operators.

## How to use

### Pre-Requisites

1. MXNet
2. Python3


```bash
# Install the version of MXNet to be tested
pip install mxnet       # For CPU (By default comes with MKLDNN)
pip install mxnet-cu10  # For GPU with CUDA 9.2

# Clone the operator benchmark library
git clone https://github.com/sandeep-krishnamurthy/dl-operator-benchmark
```


### Run benchmarks for all the operators

Below command runs all the MXNet operators (NDArray and Gluon) benchmarks with default inputs and saves the final result as JSON in the provided file.

```
python dl-operator-benchmark/run_all_mxnet_operator_benchmarks.py --output-format json --output-file mxnet_operator_benchmark_results.json

```

**Other Options:**

1. **output-format** : `json` or `md` for markdown file output or `csv`.

2. **ctx** : By default, `cpu` on CPU machine, `gpu(0)` on GPU machine. You can override and set the global context for all operator benchmarks. Example: `--ctx gpu(2)`.

3. **dtype** : By default, `float32`. You can override and set the global dtype for all operator benchmarks. Example: `--dtype float64`.

### Run benchmarks for all the operators in a specific category

For example, you want to run benchmarks for all `NDArray Arithmetic Operators`, you just run the following python script.

```python
#! /usr/bin/python
from mxnet_benchmarks.nd import run_all_arithmetic_operations_benchmarks

# Run all Arithmetic operations benchmarks with default input values
run_all_arithmetic_operations_benchmarks()

```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
MX_Add_Forward_Backward_Time - 0.015201 seconds
MX_Multiply_Forward_Backward_Time - 0.021678 seconds
MX_Subtract_Forward_Backward_Time - 0.016154 seconds
MX_Divide_Forward_Backward_Time - 0.024327 seconds
MX_Modulo_Forward_Backward_Time - 0.045726 seconds
MX_Power_Forward_Backward_Time - 0.077152 seconds
MX_Negative_Forward_Backward_Time - 0.014472 seconds
MX_Inplace_Add_Forward_Time - 0.003824 seconds
MX_Inplace_Subtract_Forward_Time - 0.004137 seconds
MX_Inplace_Multiply_Forward_Time - 0.006589 seconds
MX_Inplace_Division_Forward_Time - 0.003869 seconds
MX_Inplace_Modulo_Forward_Time - 0.018180 seconds
```

### Run benchmarks for specific operator

For example, you want to run benchmarks for `nd.add` operator in MXNet, you just run the following python script.

#### CASE 1 - Default Inputs for Operators

```python
#! /usr/bin/python
from mxnet_benchmarks.nd import Add

# Run all Arithmetic operations benchmarks with default input values
add_benchmark = Add()
add_benchmark.run_benchmark()
add_benchmark.print_benchmark_results()

```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
MX_Add_Forward_Backward_Time - 0.015201 seconds
```

#### CASE 2 - Customize Inputs for Operators

In this case, let us assume, you want to run benchmarks on a `float64` tensor instead of a default `float32`.

```python
#! /usr/bin/python
from mxnet_benchmarks.nd import Add

# Run all Arithmetic operations benchmarks with default input values
add_benchmark = Add(inputs={"dtype": "float64"})
add_benchmark.run_benchmark()
add_benchmark.print_benchmark_results()

```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
MX_Add_Forward_Backward_Time - 0.025405 seconds
```

**NOTE:** You can print the input parameters used for a benchmark as shown below.

```python
from mxnet_benchmarks.nd import Add

# Run all Arithmetic operations benchmarks with default input values
add_benchmark = Add(inputs={"dtype": "float64"})
print(add_benchmark.inputs)
```

Output:
```
{'lhs': (1024, 1024), 'rhs': (1024, 1024), 'initializer': <function normal at 0x117b607b8>, 'run_backward': True, 'dtype': 'float64'}
```

## Future Development

1. Logging
2. Currently around 134 MXNet operators (out of around 250) are supported for benchmarks. Help add more operators support.
2. Add support for Memory profiling and benchmarking.
3. Support more complex operator structure for benchmarking. Example: Fused operator - Conv + BatchNorm, Conv + Relu etc.
4. Integration with MXNet profiler to get more fine grained profiling results such as eliminate Python layer overhead, pure forward only timing, backward only timing.
5. In future, we plan to support PyTorch and other deep learning libraries to help users compare individual operator performance across frameworks.
