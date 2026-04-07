Add sigmoid, tanh, and power functions.

1. In mlpl-runtime, add built-in functions:
   - sigmoid(a) -> element-wise 1/(1+exp(-x))
   - tanh_fn(a) -> element-wise tanh (name avoids Rust keyword conflict)
   - pow(a, b) -> element-wise a^b (with scalar broadcasting)

2. sigmoid is critical for logistic regression. It must handle
   large positive/negative inputs without overflow:
   - sigmoid(100) -> ~1.0 (not infinity)
   - sigmoid(-100) -> ~0.0 (not NaN)

TDD:
- sigmoid(0) -> 0.5
- sigmoid([0, 100, -100]) -> [0.5, ~1.0, ~0.0]
- tanh_fn(0) -> 0.0
- tanh_fn([0, 1, -1]) -> [0, ~0.762, ~-0.762]
- pow(2, 3) -> 8.0
- pow([1,2,3], 2) -> [1, 4, 9] (scalar broadcast)
- pow([2,3], [3,2]) -> [8, 9] (element-wise)

Allowed: crates/mlpl-runtime/, crates/mlpl-array/, crates/mlpl-eval/
