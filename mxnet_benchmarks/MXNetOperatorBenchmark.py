import mxnet as mx
import mxnet.ndarray as nd

from abc import ABC, abstractmethod


class MXNetOperatorBenchmarkBase(ABC):
    """Abstract Base class for all MXNet operator benchmarks.
    """
    def __init__(self, ctx=mx.cpu(), warmup=10, runs=10, inputs={}):
        self.ctx = ctx
        self.runs = runs
        self.warmup = warmup
        self.results = {}
        self.inputs = inputs

    @abstractmethod
    def run_benchmark(self):
        pass

    def print_benchmark_results(self):
        if not len(self.results):
            print("No benchmark results found. Run the benchmark before printing results!")
            return

        for key, val in self.results.items():
            print("{} - {:.6f} seconds".format(key, val))
