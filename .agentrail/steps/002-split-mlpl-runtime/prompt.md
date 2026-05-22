Tech-debt saga step 002: split mlpl-runtime crate (18 modules; max 7).

Same facade pattern as step 001. Split into:

1. mlpl-runtime (core, facade) -- the dispatch registry: BUILTINS slice, call_builtin, RuntimeError, the few helpers everything else depends on.

2. mlpl-runtime-math (new) -- math + comparison + reduce builtins: gt, lt, eq, exp, log, sqrt, abs, sigmoid, tanh_fn, sum, prod, mean, max, min, argmax, argmin, etc. Pure scalar/elementwise.

3. mlpl-runtime-array (new) -- shape + indexing + linalg builtins: reshape, transpose, iota, zeros, ones, fill, concat, take, dot, matmul, broadcast_to, etc.

4. mlpl-runtime-ml (new) -- ML-shaped builtins: softmax, attention helpers, one_hot, scaled_dot_product, embed lookup, image_decode, randn, randn_normal, etc.

Process: same as step 001. lib.rs of mlpl-runtime becomes a facade. Every existing downstream import keeps working.

Target retirement: 1 Crate-FAIL + ~4-8 Module-Fn-Count FAILs.

Quality gates same as step 001: workspace tests green, clippy + fmt green, sw-checklist net-negative on BOTH fails AND warnings.