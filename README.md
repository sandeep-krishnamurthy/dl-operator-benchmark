# DL Operator Benchmarks
A Python framework for doing benchmarking for individual operators in Apache MXNet (Coming Soon - PyTorch) Deep Learning Libraries.

## Features

1. Individual operator level benchmarks to capture "Speed" - operator execution time for forward/backward/both.
2. Individual operator benchmarks to capture "Memory" usage. (Coming soon..)
3. Benchmarks for commonly fused operators. Ex: Conv + Relu, Conv + BatchNorm.
3. Benchmarks for Apache MXNet operators. (Support for PyTorch coming soon).
4. Benchmarks for operators with varying inputs to uncover any performance issues due to skewed input data. Example: Measuring operator performance on small input tensors, large input tensors along with average normally used tensor sizes.

## Motivation

Benchmarks are usually done end-to-end for a given Network Architecture. For example: ResNet-50 benchmarks on ImageNet data. This is good measurement of overall performance and health of a deep learning framework. However, it is important to note the following important factors:
1. Users use a lot more operators that are not part of a standard network like ResNet. Example: Tensor manipulation operators like mean, max, topk etc.   
2. A standard Network Architecture like ResNet-50 is made up of many operators Ex: Convolution2D, Softmax, Dense and more. Consider the following scenarios:
    1. We improved the performance of Convolution2D operator, but due to a bug, Softmax performance went down. Overall, we may observe end to end benchmarks are running fine, we may miss out the performance degradation of a single operator which can accumulate and become untraceable.
    2. You need to see in a given network, which operator is taking maximum time and plan optimization work. With end to end benchmarks, it is hard to get more fine grained numbers at operator level.
3. We need to know on different hardware infrastructure (Ex: CPU with MKLDNN, GPU with NVIDIA CUDA and cuDNN) how different operators performs. With these details, we can plan the optimization work at operator level, which exponentially boost up end to end performance.
4. You want to have nightly performance tests across all operators in a deep learning framework to catch regressions early. 

Hence, in this framework, we will build the functionality to allow users and developers of deep learning frameworks to easily run benchmarks for individual operators.

## How to use

### Pre-Requisites

1. MXNet
2. Python3


```bash
# Install the version of MXNet to be tested
pip install mxnet       # For CPU (By default comes with MKLDNN)
pip install mxnet-cu92  # For GPU with CUDA 9.2

# Clone the operator benchmark library
git clone https://github.com/sandeep-krishnamurthy/dl-operator-benchmark
```


### Run benchmarks for all the operators

```
python dl-operator-benchmark/benchmark_driver.py

```

### Run benchmarks for all the operators in a specific category

For example, you want to run benchmarks for all `Arithmetic Operators`, you just run the following python script.

```python
#! /usr/bin/python
from mxnet_benchmarks.tensor_operations.arithmetic_operations import run_all_arithmetic_operations_benchmarks

# Run all Arithmetic operations benchmarks with default input values
run_all_arithmetic_operations_benchmarks()

```

### Run benchmarks for specific operator

For example, you want to run benchmarks for `Addition` operator, you just run the following python script.

```python
#! /usr/bin/python
from mxnet_benchmarks.tensor_operations.arithmetic_operations import Add

# Run all Arithmetic operations benchmarks with default input values
add_benchmark = Add()
add_benchmark.run_benchmark()
add_benchmark.print_benchmark_results()

```