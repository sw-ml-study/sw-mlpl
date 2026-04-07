Add comparison and logical operations for ML use cases.

1. Add element-wise comparison that produces 0.0/1.0 arrays:
   - gt(a, b) -> 1.0 where a > b, else 0.0
   - lt(a, b) -> 1.0 where a < b, else 0.0
   - eq(a, b) -> 1.0 where a == b, else 0.0

   These work with scalar broadcasting like arithmetic ops.

2. Add mean(a) built-in:
   - mean(a) -> reduce_add(a) / elem_count as scalar
   - Needed for loss computation and accuracy

TDD:
- gt([3, 1, 2], [2, 2, 2]) -> [1, 0, 0]
- gt([1, 2, 3], 2) -> [0, 0, 1] (scalar broadcast)
- lt(1, 2) -> 1.0
- eq([1, 2, 3], [1, 0, 3]) -> [1, 0, 1]
- mean([2, 4, 6]) -> 4.0
- mean(reshape(iota(6), [2, 3])) -> 2.5

Allowed: crates/mlpl-runtime/, crates/mlpl-array/, crates/mlpl-eval/
