"""Performance benchmark tests for MXNet NDArray Logical Operations
1. logical_and
2. logical_or
3. logical_xor
4. logical_not

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
3. Probably we can refactor the common logic of all these binary operations into a parent
   MXNetBinaryOperatorBenchmarkBase?
"""
