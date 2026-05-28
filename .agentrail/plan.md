# Decompose mlpl-array into sparse siblings (saga 53)

Saga 52 moved mlpl-array into components/lang-core/ but left its
13 modules intact -- still a Crate Module Count FAIL. The
component migration is incomplete without the SPLIT.

Goal: decompose the 13-module mlpl-array into multiple sparse
sibling crates within components/lang-core/. Use the
extension-trait pattern so all 600+ call sites of `a.matmul(&b)`
etc. keep working with one new `use` line per file.

## Why extension traits

Operations currently live as `impl DenseArray { fn matmul(...) }`.
Splitting these to free functions in another crate breaks every
call site (`a.matmul(&b)` -> `matmul(&a, &b)`).

Extension traits keep the method syntax:
```rust
// In mlpl-array-matmul:
pub trait MatmulExt {
    fn matmul(&self, other: &DenseArray) -> Result<DenseArray, ArrayError>;
}
impl MatmulExt for DenseArray { ... }
pub mod prelude { pub use super::MatmulExt; }

// At call sites (one-line change per file):
use mlpl_array_matmul::prelude::*;
a.matmul(&b)?  // unchanged
```

## Proposed splits

After saga 53, components/lang-core/crates/ contains:
- mlpl-core: spans, identifiers, base types (unchanged)
- mlpl-array: DenseArray + Shape + dense/shape/error/display/indexing
  (5-6 modules, sparse)
- mlpl-array-ops-element: ops_binop, ops_strides as extension trait
- mlpl-array-ops-shape: ops_reshape, ops_transpose
- mlpl-array-ops-reduce: ops_reduce (reduce_axis, argmax_axis)
- mlpl-array-ops-compose: ops_concat (concat, stack, patchify, take)
- mlpl-array-ops-matmul: ops_matmul (matmul, dot)
- mlpl-eval-core (unchanged)

mlpl-array goes from 13 modules (FAIL) to ~6 (PASS).
Each new sibling has 1-2 modules and 1-4 functions (PASS).

Also opportunistically retires Function LOC warnings as we touch
each operation (matmul 50, reduce_axis 46, argmax_axis 44,
patchify 43, stack 43, apply_binop 43, take 41, concat 34,
transpose 37).

## Step plan

1. **scaffold-siblings**: create the 5 new sibling crate skeletons
   (Cargo.toml + lib.rs + prelude) and register in lang-core
   workspace. Verify empty siblings + main + others all build.
2. **split-ops-matmul**: extract ops_matmul.rs into mlpl-array-ops-matmul
   as MatmulExt + DotExt. Update mlpl-array's lib.rs to remove the
   moved file. Find and update all callers (~50 sites likely) to add
   the `use` line. Verify build + tests.
3. **split-ops-reduce**: extract ops_reduce.rs (ReduceExt, ArgmaxExt).
4. **split-ops-compose**: extract ops_concat.rs (ConcatExt, StackExt,
   PatchifyExt, TakeExt).
5. **split-ops-shape**: extract ops_reshape.rs + ops_transpose.rs
   (ReshapeExt, TransposeExt).
6. **split-ops-element**: extract ops_binop.rs + ops_strides.rs.
7. **close**: sw-checklist delta. Expect mlpl-array FAIL retired
   plus 8-10 Function LOC warnings retired.

Each step ends with green cargo check + test from all 4 workspaces
and is committed individually -- if a step fails, the tree is still
on a working revision.
