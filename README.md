# MLPL

MLPL is a Rust-first array and tensor programming language for machine learning, visualization, and experimentation.

Inspired by APL, APL2, J, and BQN.

## Quick Start

```bash
# Build
cargo build

# Interactive REPL
cargo run -p mlpl-repl

# Run a demo script
cargo run -p mlpl-repl -- -f demos/basics.mlpl
```

## What MLPL Can Do

```
mlpl> 1 + 2
3
mlpl> [1, 2, 3] * 10
10 20 30
mlpl> x = iota(12)
mlpl> m = reshape(x, [3, 4])
0 1 2 3
4 5 6 7
8 9 10 11
mlpl> transpose(m)
0 4 8
1 5 9
2 6 10
3 7 11
mlpl> reduce_add(m, 0)
12 15 18 21
mlpl> reduce_add([1, 2, 3, 4, 5])
15
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `iota(n)` | Integers 0..n |
| `shape(a)` | Dimension vector |
| `rank(a)` | Number of dimensions |
| `reshape(a, dims)` | Reshape array |
| `transpose(a)` | Reverse axis order |
| `reduce_add(a)` | Sum all elements |
| `reduce_add(a, axis)` | Sum along axis |
| `reduce_mul(a)` | Product of all elements |
| `reduce_mul(a, axis)` | Product along axis |

## Features

- Dense arrays with row-major storage
- Scalar broadcasting (scalar op array)
- Element-wise arithmetic (+, -, *, /)
- Unary negation (-x, -[1,2,3])
- Axis-specific reductions
- Variable assignment and persistence
- Execution tracing with JSON export
- REPL with :help, :trace, :clear commands
- Script file execution (-f flag)

## Architecture

Cellular monorepo with narrow crates:

`core -> array/parser -> runtime -> eval -> trace -> viz/wasm/apps -> ml`

See `docs/architecture.md` for details.

## Development

```bash
cargo test                                          # run all tests
cargo clippy --all-targets --all-features -- -D warnings  # lint
cargo fmt --all                                     # format
```

## Demo Scripts

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl        # arithmetic and arrays
cargo run -p mlpl-repl -- -f demos/matrix_ops.mlpl     # reshape, transpose
cargo run -p mlpl-repl -- -f demos/computation.mlpl    # the "42" computation
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace  # with tracing
```
