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
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace   # execution tracing
```
