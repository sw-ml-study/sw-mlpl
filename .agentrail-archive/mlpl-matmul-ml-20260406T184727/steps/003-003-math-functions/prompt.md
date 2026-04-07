Add element-wise math functions: exp, log, sqrt, abs.

1. In mlpl-array, add DenseArray::map(f: fn(f64) -> f64) -> DenseArray:
   - Applies f to every element, returns new array with same shape

2. In mlpl-runtime, add built-in functions:
   - exp(a)  -> element-wise e^x
   - log(a)  -> element-wise ln(x)
   - sqrt(a) -> element-wise square root
   - abs(a)  -> element-wise absolute value

Each takes one array argument and returns same-shape array.

TDD:
- exp(0) -> 1.0
- exp([0, 1]) -> [1.0, e]
- log(1) -> 0.0
- log(exp(2)) -> ~2.0 (round-trip)
- sqrt(4) -> 2.0
- sqrt([1, 4, 9]) -> [1, 2, 3]
- abs(-5) -> 5.0
- abs([-3, 0, 3]) -> [3, 0, 3]

Allowed: crates/mlpl-array/, crates/mlpl-runtime/, crates/mlpl-eval/
