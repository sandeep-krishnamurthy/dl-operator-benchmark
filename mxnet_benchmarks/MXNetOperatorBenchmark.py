import mxnet as mx

from abc import ABC, abstractmethod

from utils.common_utils import prepare_input_parameters


class MXNetOperatorBenchmarkBase(ABC):
    """Abstract Base class for all MXNet operator benchmarks.
    """

    def __init__(self, ctx=mx.cpu(), warmup=10, runs=10, default_parameters={}, custom_parameters=None):
        self.ctx = ctx
        self.runs = runs
        self.warmup = warmup
        self.results = {}
        self.inputs = prepare_input_parameters(caller=self.__class__.__name__,
                                               default_parameters=default_parameters,
                                               custom_parameters=custom_parameters)

    @abstractmethod
    def run_benchmark(self):
        pass

    def print_benchmark_results(self):
        if not len(self.results):
            print("No benchmark results found. Run the benchmark before printing results!")
            return

        for key, val in self.results.items():
            print("{} - {:.6f} seconds".format(key, val))

    def get_benchmark_results(self):
        return self.results
