import argparse

import mxnet as mx

from utils.common_utils import save_to_file
from mxnet_benchmarks.benchmark_executor import run_all_mxnet_operator_benchmarks


def _parse_mxnet_context(ctx):
    # TODO - Support specific device by indexing like mx.gpu(0), mx.gpu(1)
    if not ctx:
        raise ValueError("Context cannot be null or empty")

    if ctx.lower() in ['cpu', 'gpu']:
        return mx.context.Context(ctx)
    else:
        raise ValueError("Invalid context provided - %s. Supported options - cpu, gpu".format(ctx))


if __name__ == 'main':
    # CLI Parser

    # 1. GET USER INPUTS
    parser = argparse.ArgumentParser(
        description='Run all the MXNet operators (NDArray and Gluon) benchmarks with default '
                    'inputs')

    parser.add_argument('--ctx', type=str, default='cpu',
                        help='Global context to run all benchmarks. By default, cpu on a '
                             'CPU machine, gpu(0) on a GPU machine. '
                             'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
    parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
                                                                     'float32. Valid Inputs - float32, float64.')
    parser.add_argument('--output-format', type=str, default='json',
                        help='Benchmark result output format. By default, json. '
                             'Valid Inputs - json, md, csv')

    parser.add_argument('--output-file', type=str, default='./mxnet_operator_benchmarks.json',
                        help='Name and path for the '
                             'output file.')

    # TODO - Input validation
    user_options = parser.parse_args()
    print("Running MXNet operator benchmarks with the following options: ")
    print(user_options)

    # 2. RUN BENCHMARKS
    ctx = _parse_mxnet_context(user_options.ctx)
    inputs = {"dtype": user_options.dtype}
    final_benchmark_result_map = run_all_mxnet_operator_benchmarks(ctx=ctx, inputs=inputs)

    # 3. PREPARE OUTPUTS
    save_to_file(final_benchmark_result_map, user_options.output_file, user_options.output_format)
