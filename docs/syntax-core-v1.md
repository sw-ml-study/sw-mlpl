# MLPL v1 Syntax

This document defines the surface syntax for MLPL v1. It is deliberately
simple -- a small scripting language with array operations. APL/BQN-style
symbolic notation is deferred to a future Unicode alias layer.

## Numeric Literals

```
42        # integer
-3        # negative integer
1.5       # float
-0.25     # negative float
0         # zero
```

Integers are sequences of digits, optionally preceded by `-`.
Floats contain a decimal point with digits on both sides.
No scientific notation in v1.

## Identifiers

```
x
my_var
result2
_temp
```

Start with an ASCII letter or underscore. Contain ASCII letters,
digits, and underscores. Case-sensitive. No length limit.

Reserved words (v1): none. Built-in functions are looked up by name
at eval time, not reserved at the syntax level.

## Array Literals

```
[1, 2, 3]              # vector (rank 1)
[[1, 2], [3, 4]]       # matrix (rank 2)
[[1, 2, 3], [4, 5, 6]] # 2x3 matrix
```

Square brackets with comma-separated elements. Nesting creates
higher-rank arrays. All inner arrays must have the same length.

A bare number (no brackets) is a scalar.

## Operators

Binary infix operators for arithmetic:

```
x + y     # add
x - y     # subtract
x * y     # multiply
x / y     # divide
```

Precedence (high to low):
1. `*`, `/` (left-associative)
2. `+`, `-` (left-associative)
3. `=` (assignment, right-associative)

Parentheses override precedence: `(x + y) * z`

Operators apply element-wise on arrays of the same shape,
or broadcast a scalar to match the other operand's shape.

## Function Calls

```
reshape([1, 2, 3, 4], [2, 2])
transpose([[1, 2], [3, 4]])
shape(x)
iota(6)
reduce_add([1, 2, 3])
```

Named function followed by parenthesized, comma-separated arguments.
No method syntax. No anonymous functions in v1.

### Built-in functions (v1)

| Name         | Args | Description                  |
|--------------|------|------------------------------|
| `reshape`    | 2    | reshape array to new shape   |
| `transpose`  | 1    | reverse axis order           |
| `shape`      | 1    | return shape as vector       |
| `rank`       | 1    | return rank as scalar        |
| `iota`       | 1    | integers 0..n as vector      |
| `reduce_add` | 1    | sum all elements             |
| `reduce_mul` | 1    | product of all elements      |

## Assignment

```
x = [1, 2, 3]
y = reshape(x, [1, 3])
z = x + 1
```

Single `=` binds a name to a value. No `let` keyword.
Reassignment is allowed. Scope is flat (no blocks in v1).

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
Blank lines are ignored. Trailing semicolons are optional.

## Whitespace

Whitespace (spaces, tabs) is not significant except as a token
separator. Newlines are significant as statement separators.

## Examples

```
# scalar arithmetic
1 + 2

# vector arithmetic
[1, 2, 3] + [4, 5, 6]

# scalar broadcast
[1, 2, 3] * 10

# create and reshape
x = iota(12)
m = reshape(x, [3, 4])

# transpose a matrix
t = transpose(m)

# check shape
shape(t)

# reduce
reduce_add([1, 2, 3, 4, 5])

# multi-step computation
data = [1, 2, 3, 4, 5, 6]
grid = reshape(data, [2, 3])
scaled = grid * 2
result = reduce_add(scaled)
```
