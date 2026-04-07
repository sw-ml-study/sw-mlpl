# MLPL User Guide

## Getting Started

### Browser REPL

The fastest way to try MLPL is in the browser:

[https://sw-ml-study.github.io/sw-mlpl/](https://sw-ml-study.github.io/sw-mlpl/)

No installation required. Type expressions and see results instantly.

### Command-Line REPL

```bash
# Build and launch the REPL
cargo run -p mlpl-repl
```

The REPL shows a `mlpl>` prompt. Type expressions and press Enter:

```
mlpl> 1 + 2
3
mlpl> [1, 2, 3] * 10
10 20 30
```

### Running Script Files

Save MLPL code in a `.mlpl` file and run it:

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl
```

The REPL executes each line and prints results.

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help` | Show built-in function list and syntax summary |
| `:clear` | Reset all variables to start fresh |
| `:trace on` | Enable execution tracing |
| `:trace off` | Disable execution tracing |
| `:trace` | Show summary of last trace |
| `:trace json` | Print last trace as JSON |
| `:trace json <file>` | Write trace JSON to a file |
| `exit` | Quit the REPL |

## Working with Arrays

### Creating Arrays

```
# Scalars
42
-3.14

# Vectors
[1, 2, 3, 4, 5]

# Matrices (nested arrays)
[[1, 2, 3], [4, 5, 6]]

# Generate a sequence
iota(10)          # 0 1 2 3 4 5 6 7 8 9

# Construct arrays of a specific shape
zeros([3, 4])     # 3x4 matrix of zeros
ones([2, 2])      # 2x2 matrix of ones
fill([5], 3.14)   # vector of five 3.14s

# Seeded random arrays (deterministic for a given seed)
random(42, [3, 4])  # 3x4 uniform [0, 1) values
randn(42, [1000])   # 1000 standard-normal values

# Argmax (flat scalar, or along an axis)
argmax([1, 5, 2, 4])                    # scalar: 1
argmax(reshape(iota(6), [2, 3]), 1)     # per-row argmax

# Synthetic gaussian blobs dataset
# 3 centers, 20 points each -> 60x3 matrix of [x, y, label]
blobs(42, 20, [[0, 0], [3, 3], [-3, 3]])
```

### Reshaping and Transposing

```
x = iota(12)
m = reshape(x, [3, 4])
# 0  1  2  3
# 4  5  6  7
# 8  9  10 11

transpose(m)
# 0 4 8
# 1 5 9
# 2 6 10
# 3 7 11

shape(m)       # 3 4
rank(m)        # 2
```

### Reductions

```
reduce_add([1, 2, 3, 4, 5])          # 15 (sum all)
reduce_mul([1, 2, 3, 4, 5])          # 120 (product all)

m = reshape(iota(6), [2, 3])
reduce_add(m, 0)                      # sum along rows: 3 5 7
reduce_add(m, 1)                      # sum along columns: 3 12
```

## Variables

Variables persist across lines in the same REPL session:

```
x = 42
y = x + 8
y                 # 50

data = [1, 2, 3]
scaled = data * 10
scaled            # 10 20 30
```

Use `:clear` to reset all variables.

## Arithmetic and Broadcasting

All arithmetic operators work element-wise on arrays:

```
[1, 2, 3] + [4, 5, 6]     # 5 7 9
[10, 20] - [3, 7]          # 7 13
[2, 3, 4] * [5, 6, 7]      # 10 18 28
[10, 20, 30] / [2, 4, 5]   # 5 5 6
```

A scalar is broadcast to match the array shape:

```
[1, 2, 3] + 10     # 11 12 13
5 * [1, 2, 3]       # 5 10 15
```

## Linear Algebra

### Dot Product

```
a = [1, 2, 3]
b = [4, 5, 6]
dot(a, b)          # 32 (1*4 + 2*5 + 3*6)
```

### Matrix Multiplication

```
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
matmul(A, B)
# 19 22
# 43 50
```

Matrix-vector multiplication:

```
W = [[1, 2], [3, 4], [5, 6]]
x = reshape([1, 1], [2, 1])
matmul(W, x)
# 3
# 7
# 11
```

## Math Functions

All math functions apply element-wise:

```
exp(1)             # 2.718281828...
log(exp(1))        # 1
sqrt([4, 9, 16])   # 2 3 4
abs([-3, 0, 5])    # 3 0 5
pow([2, 3], [3, 2])  # 8 9
```

## ML Activations

```
sigmoid(0)         # 0.5
sigmoid([0, 1, -1])
# 0.5  0.731...  0.269...

tanh_fn([0, 1, -1])
# 0  0.762...  -0.762...
```

## Comparisons and Statistics

Comparison functions return 0 (false) or 1 (true):

```
gt([3, 1, 4], [2, 2, 2])   # 1 0 1
lt([1, 5, 3], [2, 2, 4])   # 1 0 1
eq([1, 2, 3], [1, 0, 3])   # 1 0 1
```

Compute the mean of an array:

```
mean([2, 4, 6, 8])          # 5
```

## Loops

The `repeat` construct runs a block a fixed number of times:

```
x = 0
repeat 100 { x = x + 1 }
x    # 100
```

Multiple statements in the body:

```
total = 0
count = 0
repeat 5 {
  count = count + 1;
  total = total + count
}
total    # 15
```

## Visualizing Data

The `svg(data, type)` built-in renders an array as an inline SVG
diagram. In the browser REPL the SVG is displayed directly below
the input; in the CLI REPL it prints a one-line summary and can
optionally be written to a file with `--svg-out <dir>`.

### Diagram types

```
# Scatter: Nx2 matrix of (x, y) points
svg([[0,0],[1,1],[2,4],[3,9],[4,16]], "scatter")

# Line: a vector becomes a polyline; Nx2 becomes connected (x,y) points
svg([1, 3, 2, 5, 4, 6], "line")

# Bar: one bar per element of a vector
svg([3, 1, 4, 1, 5, 9, 2, 6], "bar")

# Heatmap: MxN matrix as a colored grid (viridis ramp)
svg(reshape(iota(25), [5, 5]), "heatmap")
```

### Loss curve walkthrough

The `loss_curve.mlpl` demo fits `y = w*x` to a small dataset by
sweeping `w` over a range and computing the mean squared error at
every value:

```
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]              # true slope = 2

ws = iota(25) / 4 - 1            # 25 candidate slopes
WS = reshape(ws, [25, 1])
preds = matmul(WS, reshape(x, [1, 5]))
YS    = matmul(ones([25, 1]), reshape(y, [1, 5]))
errs  = preds - YS
losses = reduce_add(errs * errs, 1) / 5

svg(losses, "line")              # render the loss curve
```

The result is a U-shaped curve with its minimum at `w = 2`.

### Decision boundary

`svg(grid_outputs, "decision_boundary", training)` renders a
classifier's probability surface over a 2D region with the
training points overlaid:

```
gx = grid([0, 1, 0, 1], 20)      # 400 (x, y) points in the unit square
# ... train logistic regression to get w and b ...
gz = matmul(gx, reshape(w, [2, 1])) + b
gp = sigmoid(reshape(gz, [400]))
surface = reshape(gp, [20, 20])
train = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
svg(surface, "decision_boundary", train)
```

See `demos/decision_boundary.mlpl` for the full demo.

### Analysis helpers

`svg()` is a low-level primitive. For common diagrams there are
higher-level helpers that compute the right view of the data and
render a complete picture in one call:

```
hist([1, 2, 2, 3, 3, 3, 4, 4, 5], 5)
scatter_labeled([[0,0],[1,1],[0,1],[1,0]], [0, 0, 1, 1])
loss_curve([5.0, 3.0, 2.0, 1.0, 0.5, 0.25])
confusion_matrix([0,1,2,1,0], [0,1,1,1,0])
boundary_2d(grid_probs, [20, 20], training_points, training_labels)
```

`demos/analysis_demo.mlpl` walks through training a classifier and
rendering its loss curve, confusion matrix, and decision boundary
in a single script.

## Execution Tracing

Enable tracing to inspect what MLPL does internally:

```
mlpl> :trace on
Tracing enabled.
mlpl> [1, 2, 3] + [4, 5, 6]
5 7 9

mlpl> :trace
Trace for: [1, 2, 3] + [4, 5, 6]
Events: 3
  [  0] ArrayLit     span=0..9
  [  1] ArrayLit     span=12..21
  [  2] BinOp        span=0..21
```

Export the trace as JSON for external analysis:

```
mlpl> :trace json output.json
Trace written to output.json
```

Run scripts with tracing enabled:

```bash
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace
```

## Example: Logistic Regression

Train a model to learn the AND gate (output 1 only when both
inputs are 1):

```
# Dataset
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

# Initialize weights
w = zeros([2])
b = 0
lr = 1.0
n = 4

# Train for 300 steps
repeat 300 {
  z = matmul(X, reshape(w, [2, 1])) + b;
  pred = sigmoid(z);
  dz = pred - reshape(y, [4, 1]);
  dw = reshape(matmul(transpose(X), dz), [2]) / n;
  db = mean(dz);
  w = w - lr * dw;
  b = b - lr * db
}

# Check predictions
pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)

# Measure accuracy
rounded = gt(pred, 0.5)
accuracy = mean(eq(reshape(rounded, [4]), y))
accuracy    # 1 (100%)
```

## Demo Scripts

The `demos/` directory contains ready-to-run examples:

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl              # arithmetic, arrays, variables
cargo run -p mlpl-repl -- -f demos/matrix_ops.mlpl           # reshape, transpose, reductions
cargo run -p mlpl-repl -- -f demos/computation.mlpl          # multi-step computation
cargo run -p mlpl-repl -- -f demos/repeat_demo.mlpl          # loop construct
cargo run -p mlpl-repl -- -f demos/logistic_regression.mlpl  # ML training
cargo run -p mlpl-repl -- -f demos/loss_curve.mlpl           # SVG loss curve
cargo run -p mlpl-repl -- -f demos/decision_boundary.mlpl    # 2D classifier
cargo run -p mlpl-repl -- -f demos/analysis_demo.mlpl        # analysis helpers
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace   # execution tracing
```
