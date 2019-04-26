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
