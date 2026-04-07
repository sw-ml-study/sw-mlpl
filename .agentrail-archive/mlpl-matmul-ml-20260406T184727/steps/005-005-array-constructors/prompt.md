Add fill, ones, and random array constructors.

1. In mlpl-runtime, add built-in functions:
   - zeros(shape) -> array filled with 0.0 (shape is a vector of dims)
   - ones(shape) -> array filled with 1.0
   - fill(shape, value) -> array filled with given scalar value

2. These are needed for initializing weights and biases in ML.

TDD:
- zeros([3]) -> [0, 0, 0]
- zeros([2, 3]) -> 2x3 matrix of zeros
- ones([3]) -> [1, 1, 1]
- ones([2, 2]) -> 2x2 matrix of ones
- fill([3], 5) -> [5, 5, 5]
- fill([2, 2], 0.1) -> 2x2 matrix of 0.1

Allowed: crates/mlpl-runtime/, crates/mlpl-array/, crates/mlpl-eval/
