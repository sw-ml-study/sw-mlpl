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

Save MLPL code in a `.mlpl` file and run it. Both forms work:

```bash
# Explicit -f flag (back-compat).
cargo run -p mlpl-repl -- -f demos/basics.mlpl

# Positional path (saga 31 step 007 -- enables `chmod +x` +
# `#!/usr/bin/env mlpl-repl` shebang scripts).
cargo run -p mlpl-repl -- demos/basics.mlpl
```

The REPL executes each line and prints results. For the
script-as-tool story (CLI args, stdin, exit codes), see the
[Scripting in MLPL](#scripting-in-mlpl) section.

## REPL Commands

| Command | Description |
|---------|-------------|
| `:disp(M)`, `:mean(v)`, ... | Any `:builtin(args)` calls that builtin: `:disp(M)` is exactly `disp(M)` (see "Colon forms" below) |
| `:2d` | Close 3D visualization stage |
| `:3d` | Open 3D visualization stage (also Ctrl+3) |
| `:3d on` / `:3d off` | Explicit 3D on/off (case-insensitive) |
| `:3d reset` | Re-center the 3D camera |
| `:ask <prompt>` | Send the prompt to the connected Ollama model (connect mode) |
| `:builtins` | List built-in functions by category |
| `:clear` | Reset all variables, models, and session state |
| `:connect list` / `:connect set <m>` | List / pick the server's Ollama model for `:ask` |
| `:describe <name>` | Describe a variable, model, tokenizer, built-in, or REPL command |
| `:disp(expr)` | Draw expr as an ASCII box diagram -- the `disp` builtin; any builtin works as `:name(args)` |
| `:experiments` | List captured experiment runs |
| `:fns` | List your `def u:` functions (APL's `)FNS`) |
| `:help` | Show built-in function list and syntax summary |
| `:help <topic>` | Focused help: vars, models, fns, builtins, describe |
| `:history` | List recent REPL command lines (also given to `:ask` as context) |
| `:introspect` | Run all no-arg inspectors at once |
| `:list <u:name>` | Print a function back -- verbatim source, `#` comments included |
| `:models` | List bound models with layer structure |
| `:reset` | Cancel ALL in-flight work on the connected backend (y/N prompt) |
| `:status` / `:status watch` | Connected backend(s): devices, GPUs, live CPU/RAM/GPU/VRAM |
| `:tags` | List every binding's ValueTag |
| `:tokenizers` | List bound tokenizers |
| `:trace` | Show summary of last trace |
| `:trace json` | Print last trace as JSON |
| `:trace json <file>` | Write trace JSON to a file |
| `:trace off` | Disable execution tracing |
| `:trace on` | Enable execution tracing |
| `:untag <name>` | Clear a binding's auto-attached tag |
| `:upload <name>` | Open file picker; bind photo as a variable (web only) |
| `:vars` | List bound variables with shape and tag |
| `:variables` / `:functions` / `:built-ins` / `:workspace` | Long-form aliases of `:vars` / `:fns` / `:builtins` / `:wsid` |
| `:version` | sw-MLPL version + target arch |
| `:wsid` | Workspace summary |
| `exit` / `:exit` | Quit the terminal REPL (also `quit` / `:quit`) |

In the web playground the same commands work in the REPL box, and
`:<cmd> --help` prints one command's usage.

### Colon forms: commands, calls, references

The colon prefix means three distinct things:

- `:command` -- a REPL command from the table above
  (`:vars`, `:trace on`). Add `--help` after any of them
  (`:trace --help`) for its usage line.
- `:name(args)` -- a direct BUILTIN CALL: `:disp(M)` is exactly
  `disp(M)`. Any builtin works this way from the prompt.
- `:name` bare -- a builtin REFERENCE, the value handed to
  higher-order builtins: `reduce(:add, x)`, `scan(:mul, v)`.
- `:u:name` -- a USER-FUNCTION reference: the quoted form of
  your own `def u:name`. Bind it, store it in records, and
  invoke it with `call(f, args...)` (which also accepts builtin
  references -- one calling model for both).

A bare reference is not a call, so `:disp M` (space, no
parentheses) does not run `disp` -- the REPL answers with a hint
pointing at these three forms. Note that builtin calls always
need parentheses: plain `disp M` is read as a variable named
`disp` and fails with "undefined variable".

Commands that take a name accept both spellings of it:
`:describe disp` and `:describe :disp` are the same command.

## User-Defined Functions

```
def u:zscore(x) {
    "normalize x to zero mean, unit-ish variance";   # doc-string
    m = u:mean_of(x)?;
    # subtract the mean, divide by the std estimate
    d = x - fill([tally(x)], m);
    s = u:mean_of(d * d)?;
    ok(d / sqrt(s + 0.0001))
}
```

- The `u:` prefix is REQUIRED (it keeps your names from ever
  clashing with builtins -- see "The Three Kinds of Name" in the
  Language Reference). Call with `u:zscore(v)`.
- The body's last expression is the return value; `return expr`
  exits early.
- A leading string literal is the DOC-STRING: `:fns` shows it
  beside the signature and `:describe u:zscore` prints it.
- `#` comments inside a definition are KEPT: `:list u:zscore`
  prints the function back exactly as you wrote it.
- Arguments may be arrays, Results, strings, or records.
- Functions are FIRST-CLASS via references: `f = :u:zscore`
  quotes the function; `call(f, v)` invokes it; a record of
  references is a registry (`suite = {z: :u:zscore}` then
  `call(suite.z, v)`). The Result combinators `map_ok` /
  `and_then` / `or_else` take a reference first:
  `and_then(:u:validate, u:parse(x))` chains fallible steps.
- Definitions take ANNOTATIONS: `@word [record | string]` lines
  before a `def u:` attach metadata (several stack). Any word is
  preserved as data (`annotations("u:name")` reads them back;
  bare words map to 1); `@test` additionally registers the
  function as a test with optional validated fields
  (`name`, `tags`, `skip`, `expected_failure`, `timeout_ms`):

  ```
  @test {name: "addition works", tags: ["fast"]}
  def u:addition_works() { u:assert_eq(2 + 2, 4, "adds") }
  ```

  Discovery is separate from execution: `tests()` lists stable
  names in definition order (across `include` files, in splice
  order), and `test_info(name)` returns the row -- its `fn`
  field is the `:u:` reference, so a runner invokes with
  `call(test_info(name).fn)`. `timeout_ms` is recorded for the
  runner, not enforced by the evaluator.
- Fixture lifecycle: `bracket(:u:before, :u:test, :u:after)`
  guarantees the teardown hook runs whenever setup succeeded --
  after a pass, a returned `err`, or a hard error (captured as
  the structured `{kind, message}` record). Setup failure skips
  the other hooks; the test's failure stays primary, with a
  simultaneous teardown failure retained under `teardown_error`
  in the error payload.
- Typed test events: `test_event_sink(:u:report)` registers a
  callback and `emit_test_event({version: 1, kind: "test_end",
  suite: "s", name: "n", status: "passed"})` validates and
  delivers the record. Run a suite with
  `mlpl-repl --test-events events.jsonl suite.mlpl` to also get
  each event as one JSON line, on a channel separate from the
  program's own output; in connect mode the eval response
  carries the same lines in a `test_events` array.

## Control Flow and Error Handling

```
if gt(x, 0) { "positive" } else { "not positive" }
while lt(i, 10) { i = i + 1 }           # break / continue supported
r = ok(42); unwrap_or(r, 0)             # Results: ok/err/is_ok/unwrap/...
safe = try { take(v, 0, 9) } catch e { fill([1], 0) }   # e = {kind, message}
def u:f(r) { v = r?; ok(v + 1) }        # ? unwraps Ok / early-returns Err
```

`if`/`while` are expressions; Results carry failure as data;
`try/catch` demotes a hard error into the handler's value; `?`
propagates the first Err out of a `u:` function (the railway
pattern). Full guide: `docs/error-handling.md`; the "Error
Handling (two planes, two bridges)" demo walks all of it.

## Records and Results

```
rec = {kind: "index", message: "axis 3 out of range"}
rec.message                              # field access
e = err(rec)                             # the canonical error object
get_value(e)                             # [] -- zilde Option projections
get_error(err(404))                      # [404]
```

## Connect Mode (GPU server)

Start a server and open the playground FROM it so API and UI
share one origin:

```
CUDA_COMPUTE_CAP=120 cargo build -p mlpl-serve --features cuda --release
./target/release/mlpl-serve --bind 0.0.0.0:6464 --auth required --static-dir dist-pages
# browse: http://<host>:6464/sw-mlpl/   (auto-connects to its own origin)
```

No `?connect=` needed in this setup: a page with no parameter
probes its own origin and adopts it when it answers like an
mlpl-serve. `?connect=<url>` remains the override for split
UI/server setups (the server then needs the page's origin in its
`--cors-allow` list), and `?connect=off` pins the page to
browser-local eval. NOTE: `localhost` in `?connect=` means the
machine running the BROWSER. Full design: `docs/connect-topology.md`.

Connected, the GPU-tier demos light up, heavy lines run
server-side with live telemetry sparklines, train blocks stream
a live loss curve, and Life demos stream the live board
(`emit_frame`). `:status` self-tests the link. The Connect button
reports live per-backend status (CUDA / MLX / Ollama available or
unavailable), and the Ask Ollama demo enables only when the
server's configured Ollama host is actually up.

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

The fixed-name forms cover the two most common cases:

```
reduce_add([1, 2, 3, 4, 5])          # 15 (sum all)
reduce_mul([1, 2, 3, 4, 5])          # 120 (product all)

m = reshape(iota(6), [2, 3])
reduce_add(m, 0)                      # sum along rows: 3 5 7
reduce_add(m, 1)                      # sum along columns: 3 12
```

For other binops -- `:max`, `:min`, `:and`, `:or` -- use the
higher-order form `reduce(:op, x[, axis])`:

```
reduce(:max, [3, 1, 4, 1, 5, 9, 2, 6])   # 9
reduce(:min, [3, 1, 4, 1, 5, 9, 2, 6])   # 1
reduce(:add, m, 1)                        # same as reduce_add(m, 1)

# A BuiltinRef can be bound to a variable like any other value;
# user-namespace bindings do not shadow `:foo` so this is safe:
f = :max
reduce(f, [-2, 7, -3, 4])                 # 7

add = 42                                   # user var, not the builtin
reduce(:add, [1, 2, 3])                    # still 6 -- :add lives in
                                            # its own namespace
```

The first arg is a `BuiltinRef` (`:foo` syntax) -- one of the
curated names: `:add` (== `:+`), `:mul` (== `:*`), `:min`,
`:max`, `:and`, `:or`. Anything else raises a tutoring
`TypeMismatch` listing the accepted set.

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

## Including Source Files (script mode)

A script can reuse definitions from another local source file
with a top-level declaration:

```text
include "vector.mlpl"
```

- The path is one literal, relative path -- never computed,
  absolute, or conditional. `include` is only legal at the top
  level of a source file (not inside blocks or functions), and
  only in script mode: the browser session, connect mode, and
  interactive prompt answer it with a precise error instead.
- Included definitions (`u:` functions, bindings) enter the
  program at the include site, in source order.
- Resolution is sandboxed. `mlpl-repl --source-dir DIR -f
  script.mlpl` roots all includes under DIR; without the flag
  the root is the script's own directory. Escaping the root
  (absolute paths, `..`, symlinks) is rejected with the rule
  that was broken.
- Nested includes resolve relative to the INCLUDING file. A
  file loads once per program (repeats are no-ops), and an
  include cycle errors with the complete chain.
- Errors keep their location: a parse error inside an included
  file names that file and its own line and column.

## Scripting in MLPL

MLPL ships with a small set of builtins that turn `.mlpl` files
into proper Unix scripts (saga 31): output (`print` / `eprint`),
string parsing (`to_number` / `to_int` / `env`), CLI args
(`args` / `list_get` / `list_len`), control flow (`if` / `else`,
`while` / `break` / `continue`), stdin reading (`read_stdin` /
`read_stdin_lines`), and process control (`exit`, plus
automatic exit-code propagation when the script's final value
is `Err(...)`).

### A first script

The `demos/classify.mlpl` walk-through reads a numeric score
from the first CLI argument and prints a label:

```bash
mlpl-repl demos/classify.mlpl -- 42         # prints: medium
mlpl-repl demos/classify.mlpl -- 95         # prints: high
mlpl-repl demos/classify.mlpl -- 5          # prints: low
mlpl-repl demos/classify.mlpl -- 77 verbose # prints: high, with score on stderr
mlpl-repl demos/classify.mlpl -- banana     # exits 2 with parse error on stderr
mlpl-repl demos/classify.mlpl               # prints usage, exits 0
```

The script source (`demos/classify.mlpl`) is the canonical
worked example for the four critical scripting surfaces:

```mlpl
n = list_len(args())
if n - 0 {
  raw = unwrap_or(list_get(args(), 0), "0")
  parsed = to_number(raw)
  if is_ok(parsed) { 0 } else {
    eprint(err_message(parsed))
    exit(2)
  }
  score = unwrap(parsed)
  code = if gt(score, 70) { 2 } else { if gt(score, 30) { 1 } else { 0 } }
  labels = ["low", "medium", "high"]
  i = 0
  result = while gt(list_len(labels), i) {
    if i - code { 0 } else { break unwrap(list_get(labels, i)) }
    i = i + 1
  }
  print(result)
  ok(result)
} else {
  print("usage: classify.mlpl -- SCORE [verbose]")
  ok("no-input")
}
```

### The `--` argument separator

Everything after `--` on the command line becomes the script's
own CLI args, visible via `args()`. This keeps mlpl-repl's
flags (`-f`, `--trace`, `--svg-out`, ...) cleanly separated
from the script's:

```bash
mlpl-repl my_script.mlpl --trace -- foo bar baz
#                       ^^^^^^^^    ^^^^^^^^^^^
#                       repl flag    script args -> args() = ["foo", "bar", "baz"]
```

### Shebang scripts

Because `#` already begins a line comment in MLPL, a leading
`#!` line is silently skipped by the lexer. Combine that with
the positional script-path form to get true Unix-style
executables:

```mlpl
#!/usr/bin/env mlpl-repl
print("hello from a shebang script")
```

```bash
chmod +x hello.mlpl
./hello.mlpl              # runs as a normal command
./hello.mlpl arg1 arg2    # args() = ["arg1", "arg2"]
```

### Exit codes

`mlpl-repl` in `-f` (or positional-path) mode maps the
script's final value to a Unix exit code:

| Final value | Exit code | Notes |
|-------------|-----------|-------|
| `Ok(_)` | `0` | success |
| `Err(msg)` | `1` | `msg` written to stderr |
| any non-`Result` (scalar, vector, etc.) | `0` | success |
| parse / eval failure | `1` | source line + error written to stderr |
| `exit(code)` | `code` | short-circuits everything above; `code` must be `0..=255` |

Compose with `&&` / `||` in the shell:

```bash
mlpl-repl check.mlpl && echo "ok"
mlpl-repl maybe-fail.mlpl || echo "failed: $?"
echo "1 2 3" | mlpl-repl sum-stdin.mlpl
```

### Reading stdin

`read_stdin()` returns all stdin bytes to EOF as a string;
`read_stdin_lines()` returns a `StrList` (trailing newline
stripped). Both refuse to read from an interactive terminal --
they return `Err("...stdin is a terminal; pipe input or use
args() instead")` so the REPL never hangs on a stray
`read_stdin()` at the prompt.

```bash
echo "hello" | mlpl-repl -f greet.mlpl
# where greet.mlpl is:
#   text = read_stdin()
#   print("you said:")
#   print(text)
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

## Labeled Axes

Annotation syntax on assignment attaches axis names as metadata.
Labels propagate through elementwise ops, matmul, reductions, and
activations; a mismatch surfaces as a structured error that names
both shapes:

```
X : [batch, feat] = randn(7, [60, 2])
labels(X)                         # "batch,feat"
reduce_add(X, "feat")             # reduce by axis name
```

See `docs/lang-reference.md` under "Labeled Axes" for `label`,
`relabel`, `reshape_labeled`, and `labels`.

## Autograd, Optimizers, and the Training Loop

Declare a trainable leaf with `param[shape]`. `grad(loss_expr, W)`
lifts the expression onto a reverse-mode tape and returns the
gradient with the same shape as `W`. `adam` / `momentum_sgd` take
either a single param, a list, or a model identifier:

```
W = param[1]
W = randn(1, [1]) * 2             # initialize

train 50 {
  adam(sum(W*W), W, 0.1, 0.9, 0.999, 0.00000001);
  reduce_add(W*W)
}
loss_curve(last_losses)
```

`train N { body }` mirrors `repeat` but also binds the iteration
index to `step` and captures each iteration's final value into a
`last_losses` vector.

## Model DSL

Stack layers as data. `chain(a, b, ...)` composes sequentially;
`residual(block)` adds a skip connection; `apply(m, X)` runs the
forward pass and is differentiable through every owned parameter:

```
mdl = chain(linear(2, 8, 11), tanh_layer(), linear(8, 2, 12))
X : [batch, feat] = matmul(moons(7, 60, 0.08), [[1,0],[0,1],[0,0]])
Y = one_hot(reshape(matmul(moons(7, 60, 0.08), [[0],[0],[1]]), [120]), 2)

train 100 {
  adam(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)),
       mdl, 0.05, 0.9, 0.999, 0.00000001);
  mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y))
}
```

Available layers: `linear`, `tanh_layer`, `relu_layer`,
`softmax_layer`, `rms_norm`, `attention`, `causal_attention`,
`embed`, plus `sinusoidal_encoding` for additive positional tables.

## Loading Data

The terminal REPL reads files under a sandbox (`--data-dir <path>`);
the web REPL uses a compiled-in corpus registry instead:

```
# Terminal REPL: cargo run -p mlpl-repl -- --data-dir ./data
text = load("corpus.txt")         # whole-file Value::Str
points = load("points.csv")       # numeric array, header -> labels

# Either REPL:
text = load_preloaded("tiny_corpus")
text = load_preloaded("tiny_shakespeare_snippet")
```

Dataset ops prepare training data without leaving MLPL:

```
data = reshape(iota(12), [6, 2])
s = shuffle(data, 7)
batched = batch(iota(5), 2)        # zero-pads the short tail
mask = batch_mask(iota(5), 2)       # 1 for real rows, 0 for padding
trset = split(iota(10), 0.8, 42)
vaset = val_split(iota(10), 0.8, 42)

for row in reshape(iota(6), [3, 2]) { reduce_add(row) }
last_rows                          # [1, 5, 9]
```

## Tokenizers

Byte-level tokenization is the deterministic baseline; byte-pair
encoding adds a trained merge table on top. Round-trip is lossless
for any UTF-8 input:

```
tokenize_bytes("hello")            # [104, 101, 108, 108, 111]
decode_bytes(tokenize_bytes("round trip"))

bpe = train_bpe("abababab", 260, 7)
apply_tokenizer(bpe, "abababab")
decode(bpe, apply_tokenizer(bpe, "unseen text"))
```

## Experiment Tracking

Wrap a block in `experiment "name" { ... }` to capture every scalar
assigned to a name ending in `_metric` along with the shapes of any
`param` bindings. The terminal REPL additionally writes the record
to `<--exp-dir>/<name>/<timestamp>/run.json`:

```
experiment "baseline" { loss_metric = 0.5; accuracy_metric = 0.82 }
experiment "tweak"    { loss_metric = 0.3; accuracy_metric = 0.91 }
:experiments
compare("baseline", "tweak")
```

## Training a Tiny Language Model

Saga 13 ties everything above together. `embed(V, d, seed)` is a
learned lookup table; `sinusoidal_encoding(T, d)` is deterministic
positional info; `causal_attention` masks the pre-softmax scores so
position `t` cannot peek at `t+1`; `cross_entropy(logits, targets)`
is a numerically-stable fused loss; `sample` + `top_k` plus the
`last_row` / `concat` helpers give you a generation loop:

```
corpus = load_preloaded("tiny_corpus")
tok    = train_bpe(corpus, 260, 0)
ids    = apply_tokenizer(tok, corpus)
X_all  = shift_pairs_x(ids, 8)
Y_all  = shift_pairs_y(ids, 8)
X      = reshape(X_all, [reduce_mul(shape(X_all))])
Y      = reshape(Y_all, [reduce_mul(shape(Y_all))])

V = 260 ; d = 16 ; h = 1
model = chain(embed(V, d, 0),
              causal_attention(d, h, 1),
              rms_norm(d),
              linear(d, V, 2))

experiment "tutorial_tiny_lm" {
  train 30 {
    adam(cross_entropy(apply(model, X), Y),
         model, 0.01, 0.9, 0.999, 0.00000001);
    loss_metric = cross_entropy(apply(model, X), Y)
  }
}
loss_curve(last_losses)

# Generation
prompt = apply_tokenizer(tok, "the ")
seq    = prompt
repeat 20 {
  logits = apply(model, seq);
  last   = last_row(logits);
  nxt    = sample(top_k(last, 20), 0.8, step);
  seq    = concat(seq, nxt)
}
decode(tok, seq)

# Attention heatmap
viz_ids = apply_tokenizer(tok, "the quick")
svg(attention_weights(model, viz_ids), "heatmap")
```

The web REPL's "Training and Generating" tutorial lesson walks
through the same flow interactively. `demos/tiny_lm.mlpl` and
`demos/tiny_lm_generate.mlpl` are the full-size versions (280-vocab
BPE, Shakespeare corpus, residual transformer block).

## Compiling MLPL to a Native Binary

`mlpl-build` takes a `.mlpl` script and produces a self-contained
native binary that only links against `mlpl-rt`. The compiled
program has no interpreter, no parser, and no runtime dispatch --
startup is just the OS loading an executable.

```bash
cargo run -p mlpl-build -- examples/compile-cli/hello.mlpl -o hello
./hello
# -> 42

# Cross-compile the same source to WASM
cargo run -p mlpl-build -- examples/compile-cli/hello.mlpl \
    --target wasm32-unknown-unknown -o hello.wasm
```

See `examples/compile-cli/README.md` for a complete walkthrough,
and `docs/compiling-mlpl.md` for the three-way comparison of the
interpreter, the `mlpl!` proc macro, and the `mlpl build` path.

## Demo Scripts

The `demos/` directory contains ready-to-run examples:

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl               # arithmetic, arrays, variables
cargo run -p mlpl-repl -- -f demos/matrix_ops.mlpl            # reshape, transpose, reductions
cargo run -p mlpl-repl -- -f demos/computation.mlpl           # multi-step computation
cargo run -p mlpl-repl -- -f demos/repeat_demo.mlpl           # loop construct
cargo run -p mlpl-repl -- -f demos/logistic_regression.mlpl   # ML training
cargo run -p mlpl-repl -- -f demos/loss_curve.mlpl            # SVG loss curve
cargo run -p mlpl-repl -- -f demos/decision_boundary.mlpl     # 2D classifier
cargo run -p mlpl-repl -- -f demos/analysis_demo.mlpl         # analysis helpers
cargo run -p mlpl-repl -- -f demos/kmeans.mlpl                # K-Means clustering
cargo run -p mlpl-repl -- -f demos/pca.mlpl                   # PCA via power iteration
cargo run -p mlpl-repl -- -f demos/softmax_classifier.mlpl    # 3-class softmax
cargo run -p mlpl-repl -- -f demos/tiny_mlp.mlpl              # 2-8-2 MLP on XOR-like data
cargo run -p mlpl-repl -- -f demos/moons_mlp.mlpl             # chain + train + adam on moons
cargo run -p mlpl-repl -- -f demos/circles_mlp.mlpl           # same, on circles
cargo run -p mlpl-repl -- -f demos/attention.mlpl             # Q K^T / sqrt(d) pattern
cargo run -p mlpl-repl -- -f demos/transformer_block.mlpl     # residual attention + MLP
cargo run -p mlpl-repl -- -f demos/tiny_lm.mlpl               # tiny LM training (Saga 13)
cargo run -p mlpl-repl -- -f demos/tiny_lm_generate.mlpl      # training + generation + attention heatmap
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace    # execution tracing
```
