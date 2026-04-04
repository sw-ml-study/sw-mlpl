# Array Contract

## Purpose

Define the behavioral spec for dense arrays and tensors in MLPL.
This contract governs `mlpl-array`, which is the core data structure
crate for the language.

## Key Types and Concepts

### Shape

An ordered list of non-negative dimension sizes.

- `Shape::new(dims: Vec<usize>)` -- create from dimension list
- `Shape::scalar()` -- zero-rank shape (empty dimension list)
- `shape.rank()` -- number of dimensions
- `shape.dims()` -- borrow the dimension slice
- `shape.elem_count()` -- product of all dimensions (1 for scalar)

### DenseArray

A flat-storage array with a shape.

- `DenseArray<T>` stores elements in row-major (C) order
- Created from a shape and a data vector
- Indexed by a flat offset or a multi-dimensional index
- Element type `T` is generic but initially `f64`

### Operations

- **reshape**: change shape without changing data or element order
- **transpose**: reverse dimension order, reorder data accordingly
- **indexing**: convert multi-dim index to flat offset and retrieve

## Invariants

- Shape dimension order matters: `[2, 3]` is not `[3, 2]`
- `rank == shape.len()`
- `elem_count == shape.iter().product()` (1 for scalar, empty shape)
- `DenseArray` data length must equal `shape.elem_count()`
- Reshape preserves element order (row-major layout unchanged)
- Reshape succeeds only when source and target element counts match
- Transpose of a rank-N array reverses the axis order
- Multi-dim index must have exactly `rank` components
- Each index component must be `< shape[axis]`

## Error Cases

Use explicit error variants, not string errors.

- `ShapeMismatch` -- reshape target count differs from source count
- `IndexOutOfBounds` -- index component >= dimension size
- `RankMismatch` -- index length != array rank
- `EmptyShape` -- operation requires non-empty shape but got empty
  (if zero-dim arrays are disallowed in a given context)
- `DataLengthMismatch` -- data vec length != shape elem_count

## What This Contract Does NOT Cover

- Sparse arrays or compressed storage
- Array views, slices, or strides (future)
- Broadcasting (future, likely in runtime or eval)
- GPU or SIMD acceleration
- Non-numeric element types

## Open Questions

- Whether zero-size dimensions (e.g., `[2, 0, 3]`) are allowed
- Whether shapes should use `usize` or a narrower type like `u32`
- Whether scalar is represented as rank-0 or rank-1 with shape `[1]`
- When array views and stride-based indexing enter the picture
