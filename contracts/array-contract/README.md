# Array Contract

## Purpose

Define the behavioral spec for dense arrays and tensors in MLPL.
This contract governs `mlpl-array`, which is the core data structure
crate for the language. All array operations in MLPL ultimately
resolve to operations on types defined here.

---

## 1. Shape Model

A **shape** is an ordered list of non-negative dimension sizes.

### Rank terminology

| Rank | Name   | Shape example | Element count |
|------|--------|---------------|---------------|
| 0    | scalar | `[]`          | 1             |
| 1    | vector | `[5]`         | 5             |
| 2    | matrix | `[2, 3]`      | 6             |
| 3+   | tensor | `[2, 3, 4]`   | 24            |

### Shape API

- `Shape::new(dims: Vec<usize>)` -- create from dimension list
- `Shape::scalar()` -- rank-0 shape (empty dimension list, 1 element)
- `Shape::vector(n: usize)` -- rank-1 shape `[n]`
- `shape.rank() -> usize` -- number of dimensions (`dims.len()`)
- `shape.dims() -> &[usize]` -- borrow the dimension slice
- `shape.elem_count() -> usize` -- product of all dimensions
  - scalar (empty dims): returns 1
  - contains a zero dim: returns 0

### Shape invariants

- Dimension order matters: `Shape([2, 3])` != `Shape([3, 2])`
- `rank == dims.len()`
- `elem_count == dims.iter().product::<usize>()` (1 when dims is empty)
- Shapes are value types: equality is structural (same dims = same shape)
- Shapes are immutable once created

### Shape decisions

- Scalar is rank-0 with empty dimension list, NOT rank-1 with `[1]`
- Zero-size dimensions (e.g., `[2, 0, 3]`) are allowed; they produce
  elem_count = 0 and an empty data buffer
- Dimension type is `usize`

---

## 2. Dense Storage

### Layout

- Row-major (C order) contiguous storage in a flat `Vec<T>`
- Data length must equal `shape.elem_count()`
- For a matrix with shape `[r, c]`, element at row `i`, col `j`
  is stored at flat index `i * c + j`

### Element types

- MVP element type: `f64`
- `DenseArray<T>` is generic over `T` to allow future extension
- `T` must implement `Clone` (and `Debug` for display)

### Stride model

- Strides are derived from shape, not stored independently
- For shape `[d0, d1, ..., dN]`, stride for axis `k` is
  `d(k+1) * d(k+2) * ... * dN`
- Rightmost axis has stride 1
- This is implicit in row-major layout; no explicit stride field for MVP

### DenseArray API

- `DenseArray::new(shape: Shape, data: Vec<T>) -> Result<Self, ArrayError>`
  - Fails if `data.len() != shape.elem_count()`
- `DenseArray::zeros(shape: Shape) -> Self` (where T: Default)
- `array.shape() -> &Shape`
- `array.data() -> &[T]`
- `array.rank() -> usize`
- `array.elem_count() -> usize`

---

## 3. Indexing

### Origin

- 0-origin throughout (first element is index 0)

### Multi-dimensional indexing

- `array.get(index: &[usize]) -> Result<&T, ArrayError>`
- `array.get_mut(index: &[usize]) -> Result<&mut T, ArrayError>`
- Index must have exactly `rank` components
- Each component must be `< shape[axis]`

### Flat indexing

- `array.get_flat(offset: usize) -> Result<&T, ArrayError>`
- Offset must be `< elem_count`

### Index-to-offset conversion

For shape `[d0, d1, ..., dN]` and index `[i0, i1, ..., iN]`:

```
offset = i0 * (d1 * d2 * ... * dN)
       + i1 * (d2 * ... * dN)
       + ...
       + iN
```

### Indexing invariants

- Bounds checking is always performed (no unchecked access in MVP)
- Out-of-bounds access returns an error, never panics
- An empty array (elem_count = 0) cannot be indexed

---

## 4. Reshape

### Semantics

- Changes the shape without changing the data or element order
- The underlying flat data buffer is unchanged
- Only the shape metadata changes

### Rules

- `array.reshape(new_shape: Shape) -> Result<DenseArray<T>, ArrayError>`
- Succeeds only when `new_shape.elem_count() == array.elem_count()`
- Fails with `ShapeMismatch` when counts differ
- Element ordering is preserved (row-major interpretation changes,
  but flat buffer stays the same)

### Examples

```
[1, 2, 3, 4, 5, 6] with shape [6]
  reshape to [2, 3] ->
    [[1, 2, 3],
     [4, 5, 6]]

  reshape to [3, 2] ->
    [[1, 2],
     [3, 4],
     [5, 6]]

  reshape to [2, 4] -> ERROR: ShapeMismatch (6 != 8)
```

---

## 5. Transpose

### Semantics

- Reverses the axis order
- For a matrix: swaps rows and columns
- Data is physically reordered (not a view)

### Rules

- `array.transpose() -> DenseArray<T>`
- New shape is the reverse of the original shape
- For shape `[a, b, c]`, transposed shape is `[c, b, a]`
- Element at index `[i, j, k]` in original appears at `[k, j, i]`
  in transposed array

### Special cases

- Scalar transpose: returns a copy (no-op)
- Vector transpose: returns a copy (rank-1 reversal is identity)
- Matrix transpose: classic row/column swap

### Examples

```
shape [2, 3] data [1, 2, 3, 4, 5, 6]
  -> matrix:
     [[1, 2, 3],
      [4, 5, 6]]
  -> transposed shape [3, 2]:
     [[1, 4],
      [2, 5],
      [3, 6]]
  -> transposed data [1, 4, 2, 5, 3, 6]
```

---

## 6. Broadcasting / Pervasion (high-level)

Broadcasting defines how operations apply to arrays of different shapes.
Detailed rules will be specified when element-wise operations are
implemented (likely in `mlpl-runtime`). This section establishes the
principles that `mlpl-array` must support.

### Principles

1. **Scalar extends to any shape**: a scalar operand is treated as if
   it has the same shape as the other operand, with every element equal
   to the scalar value
2. **Matching shapes**: arrays with identical shapes operate element-wise
3. **Incompatible shapes**: produce an explicit error

### Future extensions (not in MVP)

- Trailing-dimension broadcasting (NumPy-style)
- Leading-axis agreement (APL/J-style)
- The choice between these will be made during runtime contract work

### What mlpl-array provides for broadcasting

- `Shape::is_broadcast_compatible(other: &Shape) -> bool` (future)
- `Shape::broadcast_result(other: &Shape) -> Result<Shape, ArrayError>` (future)
- For MVP, only scalar-extends-to-any and same-shape are needed

---

## 7. Error Cases

All errors use explicit enum variants in `ArrayError`. No string errors.

| Variant | When | Fields |
|---------|------|--------|
| `DataLengthMismatch` | `DenseArray::new()` data length != shape elem_count | `expected: usize, got: usize` |
| `ShapeMismatch` | reshape target elem_count != source elem_count | `source: usize, target: usize` |
| `IndexOutOfBounds` | index component >= dimension size | `axis: usize, index: usize, size: usize` |
| `RankMismatch` | index length != array rank | `expected: usize, got: usize` |
| `EmptyArray` | indexing an array with elem_count = 0 | (none) |

### Error invariants

- Every error variant includes enough context to produce a useful message
- Errors implement `std::fmt::Display` with human-readable messages
- Errors implement `std::error::Error`
- No panics on invalid input -- always return `Result`

---

## 8. What This Contract Does NOT Cover

- **Parser syntax** for array literals (governed by parser-contract)
- **Evaluator semantics** for expressions (governed by eval-contract)
- **Trace serialization** of array values (governed by trace-contract)
- **Boxed/nested arrays** where elements are themselves arrays (post-MVP)
- **Sparse arrays** or compressed storage formats
- **Array views** or borrowed slices with strides (future)
- **GPU or SIMD** acceleration
- **Non-numeric element types** (strings, booleans as array elements)
- **Sorting, searching, or set operations** on arrays

---

## 9. Implementation Order

The recommended implementation sequence for `mlpl-array`:

1. `Shape` type with constructors, rank, dims, elem_count
2. `ArrayError` enum with all variants
3. `DenseArray<T>` with new, zeros, shape, data, rank, elem_count
4. Multi-dimensional indexing (get, get_mut)
5. Flat indexing (get_flat)
6. Reshape
7. Transpose
8. Display/Debug formatting

Each step should be driven by tests written first (TDD).
