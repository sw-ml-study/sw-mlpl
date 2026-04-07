# MLPL Language Reference

## Numeric Literals

```
42        # integer
-3        # negative integer
1.5       # float
-0.25     # negative float
```

Integers are sequences of digits. Floats contain a decimal point with
digits on both sides. No scientific notation.

## String Literals

```
"scatter"
"line"
"hello"
```

Double-quoted ASCII strings. Strings are a separate value kind from
arrays and are used as type-name arguments to built-ins like `svg()`.
Strings cannot be combined with numeric operators.

## Identifiers

```
x
my_var
result2
_temp
```

Start with an ASCII letter or underscore. Contain ASCII letters,
digits, and underscores. Case-sensitive.

## Array Literals

```
[1, 2, 3]              # vector (rank 1)
[[1, 2], [3, 4]]       # matrix (rank 2)
[[1, 2, 3], [4, 5, 6]] # 2x3 matrix
```

Square brackets with comma-separated elements. Nesting creates
higher-rank arrays. All inner arrays must have the same length.
A bare number (no brackets) is a scalar (rank 0).

## Operators

Binary infix operators for element-wise arithmetic:

| Operator | Description |
|----------|-------------|
| `+` | Add |
| `-` | Subtract |
| `*` | Multiply |
| `/` | Divide |
| `-x` | Unary negation |

Precedence (high to low):

1. Unary `-` (prefix negation)
2. `*`, `/` (left-associative)
3. `+`, `-` (left-associative)
4. `=` (assignment, right-associative)

Parentheses override precedence: `(x + y) * z`

Operators apply element-wise on arrays of the same shape, or
broadcast a scalar to match the other operand's shape:

```
[1, 2, 3] + [4, 5, 6]    # [5, 7, 9]
[1, 2, 3] * 10            # [10, 20, 30]
```

## Assignment

```
x = [1, 2, 3]
y = reshape(x, [1, 3])
z = x + 1
```

Single `=` binds a name to a value. No `let` keyword. Reassignment
is allowed. Scope is flat. Assignment returns the assigned value.

## Function Calls

```
reshape([1, 2, 3, 4], [2, 2])
iota(6)
reduce_add([1, 2, 3])
matmul(A, B)
```

Named function followed by parenthesized, comma-separated arguments.

## Loop Construct

```
repeat N { body }
```

Execute the body N times. N must evaluate to a non-negative integer.
The body can contain multiple statements separated by semicolons or
newlines. Returns the result of the last expression in the final
iteration (or scalar 0 if N is 0).

```
x = 0
repeat 10 { x = x + 1 }
# x is now 10
```

## Comments

```
# this is a comment
x = 42  # inline comment
```

`#` starts a line comment that runs to end of line.

## Statement Separation

```
x = 1; y = 2        # semicolon separates
x = 1
y = 2                # newline separates
```

Newlines and semicolons are both statement separators.

## Built-in Functions

### Array Operations

| Function | Args | Description |
|----------|------|-------------|
| `iota(n)` | 1 | Integers 0, 1, ..., n-1 as a vector |
| `shape(a)` | 1 | Dimension vector of array |
| `rank(a)` | 1 | Number of dimensions (scalar) |
| `reshape(a, dims)` | 2 | Reshape array to new dimensions |
| `transpose(a)` | 1 | Reverse axis order |
| `reduce_add(a)` | 1 | Sum all elements |
| `reduce_add(a, axis)` | 2 | Sum along a specific axis |
| `reduce_mul(a)` | 1 | Product of all elements |
| `reduce_mul(a, axis)` | 2 | Product along a specific axis |

### Linear Algebra

| Function | Args | Description |
|----------|------|-------------|
| `dot(a, b)` | 2 | Dot product of two vectors |
| `matmul(a, b)` | 2 | Matrix multiplication |

### Math Functions

| Function | Args | Description |
|----------|------|-------------|
| `exp(a)` | 1 | Element-wise e^x |
| `log(a)` | 1 | Element-wise natural logarithm |
| `sqrt(a)` | 1 | Element-wise square root |
| `abs(a)` | 1 | Element-wise absolute value |
| `pow(a, b)` | 2 | Element-wise a^b |

### ML Activations

| Function | Args | Description |
|----------|------|-------------|
| `sigmoid(a)` | 1 | Logistic sigmoid: 1 / (1 + exp(-x)) |
| `tanh_fn(a)` | 1 | Hyperbolic tangent |

### Comparison and Statistics

| Function | Args | Description |
|----------|------|-------------|
| `gt(a, b)` | 2 | Element-wise greater-than (returns 0 or 1) |
| `lt(a, b)` | 2 | Element-wise less-than (returns 0 or 1) |
| `eq(a, b)` | 2 | Element-wise equality (returns 0 or 1) |
| `mean(a)` | 1 | Arithmetic mean of all elements |

### Array Constructors

| Function | Args | Description |
|----------|------|-------------|
| `zeros(shape)` | 1 | Array of zeros with given shape |
| `ones(shape)` | 1 | Array of ones with given shape |
| `fill(shape, v)` | 2 | Array filled with value v |
| `grid(bounds, n)` | 2 | n*n by 2 matrix of (x, y) points over [xmin,xmax] x [ymin,ymax]; bounds is [xmin, xmax, ymin, ymax] |
| `random(seed, shape)` | 2 | Seeded uniform [0, 1) array with the given shape. `seed` is a scalar integer; `shape` is a vector of dimensions. Deterministic for a given seed. |
| `randn(seed, shape)` | 2 | Seeded standard-normal array (mean 0, variance 1), same shape semantics as `random`. Implemented via Box-Muller on the same xorshift64 stream. |
| `argmax(a)` | 1 | Scalar index of the maximum element over all elements of `a` (flat). |
| `argmax(a, axis)` | 2 | Index (as f64) of the max along `axis`; output rank is one less than input. Ties go to the first occurrence. |
| `blobs(seed, n_per_class, centers)` | 3 | Seeded 2D gaussian-blob dataset. `centers` is a Kx2 matrix (or length-2K vector) of cluster centers; returns an Nx3 matrix where each row is `[x, y, label]`, with `N = K * n_per_class` and noise sigma 0.15. |

### Visualization

| Function | Args | Description |
|----------|------|-------------|
| `svg(data, type)` | 2 | Render `data` as an SVG diagram of the given type and return the SVG string |
| `svg(data, type, aux)` | 3 | Same, with an auxiliary array (used by `decision_boundary`) |

Supported `type` values:

- `"scatter"` -- expects an Nx2 matrix; one circle per row.
- `"line"` -- a vector becomes a polyline; an Nx2 matrix becomes (x,y) points connected by lines.
- `"bar"` -- a vector becomes a bar chart with one bar per element.
- `"heatmap"` -- an MxN matrix rendered as a viridis-colored grid.
- `"decision_boundary"` -- a 2D classifier-output grid rendered as a diverging-color surface, with the third argument as an Nx3 `[x, y, label]` training matrix overlaid as colored points.

The browser REPL detects SVG return values and renders them inline.
The CLI REPL prints a `[svg: N bytes]` summary; pass `--svg-out <dir>`
to write each SVG to a file.

### Analysis helpers

High-level helpers that compute and render a complete diagram in
one call. Each returns an SVG string just like `svg()`.

| Function | Args | Description |
|----------|------|-------------|
| `hist(data, bins)` | 2 | Histogram of a vector with `bins` equal-width bins, rendered as a bar chart |
| `scatter_labeled(points, labels)` | 2 | Nx2 points colored by a length-N cluster-id vector |
| `loss_curve(losses)` | 1 | Vector of losses rendered as a line plot with axis labels |
| `confusion_matrix(predicted, actual)` | 2 | KxK heatmap of class-id predictions vs actual labels with cell counts overlaid |
| `boundary_2d(grid_outputs, dims, points, labels)` | 4 | Render a 2D classifier surface from a length-(rows*cols) vector and `[rows, cols]` dims, with separately-supplied training points and labels |

## Array Display

Arrays are displayed in a row-major layout:

```
mlpl> 42
42
mlpl> [1, 2, 3]
1 2 3
mlpl> reshape(iota(6), [2, 3])
0 1 2
3 4 5
```

Scalars print as a single number. Vectors print space-separated on
one line. Matrices print one row per line.

## Broadcasting Rules

When an operator combines a scalar with an array, the scalar is
broadcast to match the array's shape:

```
[1, 2, 3] + 10     # [11, 12, 13]
5 * [1, 2, 3]       # [5, 10, 15]
```

When both operands are arrays, they must have the same shape.

## Error Handling

Errors are reported inline with descriptive messages:

- Shape mismatches: "shapes do not match: [2, 3] vs [3, 2]"
- Unknown functions: "unknown function: foo"
- Arity errors: "reshape expects 2 arguments, got 1"
- Undefined variables: "undefined variable: x"
