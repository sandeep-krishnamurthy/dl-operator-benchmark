# DL Operator Benchmarks
A Python framework for doing benchmarking for individual operators in Apache MXNet and PyTorch Deep Learning Libraries.

## Features

1. Individual operator level benchmarks to capture - time for operator execution (speed), memory usage.
2. Fine grained individual operator level benchmarks to capture - time for forward pass, time for backward pass.
3. Benchmarks for commonly fused operators. Ex: Conv + Relu, Conv + BatchNorm.
3. Benchmarks for Apache MXNet operators. (Support for PyTorch coming soon).
4. Benchmarks for operators with varying inputs to uncover any performance issues due to skewed input data. Ex: Measuring operator performance on small input tensors, large input tensors along with average normally used tensor sizes.

## Motivation

Benchmarks are usually done end-to-end for a given Network Architecture. For example: ResNet-50 benchmarks on ImageNet data. This is good measurement of overall performance and health of a deep learning framework. However, it is important to note that a Network Architecture like ResNet-50 is made up of many operators Ex: Convolution2D, Softmax, Dense and more. Consider the following scenarios:
1. We improved the performance of Convolution2D operator, but due to a bug, Softmax performance went down. Overall, we may observe end to end benchmarks are running fine, we may miss out the performance degradation of a single operator which can accumulate and become untraceable.
2. You need to see in a given network, which operator is taking maximum time and plan optimization work. With end to end benchmarks, it is hard to get more fine grained numbers at operator level.
3. We need to know on different hardware infrastructure (Ex: CPU with MKLDNN, GPU with NVIDIA CUDA and cuDNN) how different operators performs. With these details, we can plan the optimization work at operator level, which exponentially boost up end to end performance.
4. You want to have nightly performance tests across all operators in a deep learning framework to catch regressions early. 

Hence, in this framework, we will build the functionality to allow users and developers of deep learning frameworks to easily run benchmarks for individual operators.
