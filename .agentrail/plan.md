# Scripting cluster saga

## Why this exists

MLPL today is a *demo language*: every program hardcodes its inputs,
no program takes arguments, and a script's only output is its final
expression. The audit (docs/language-audit.md findings #22-#30)
identifies a small set of additions that together turn MLPL into a
real scripting language. The critical four are joined at the hip:
landing any three without the fourth still leaves a user blocked
from writing useful scripts. They are treated as one saga here.

- **#22 (`if` / `else`).** No surface conditional expression. Today
  the only branching mechanism is `unwrap_or(r, default)` on a
  Result, plus arithmetic masks on tensor data. A script that wants
  to choose between two paths based on a scalar flag has no surface.
- **#24 (CLI arguments).** `mlpl-repl -f script.mlpl arg1 arg2`
  silently drops the trailing args. There is no `args()` builtin,
  no `ARGV`, no positional-binding syntax.
- **#26 (string-to-number).** No `to_number(s)` / `to_int(s)`. Even
  if args() landed, the script could not turn `"42"` into `42`.
- **#28 (`print`).** A script's value is the last expression only;
  intermediate values silently drop. There is no `print(value)`.

The nice-to-have findings (#23 while / break, #25 env(), #27 stdin,
#29 exit codes, #30 example demo) round out the saga but the
critical four are the minimum surface to ship.

## Goals

- **Correctness.** Each new construct passes its own focused tests;
  the existing demo set (saga 29 era) keeps passing without rewrite.
- **Ergonomics.** A 10-line MLPL script that takes a filename
  argument, loads an image, branches on a flag, and prints a label
  is achievable after this saga ships.
- **Compatibility.** The REPL keeps working unchanged for users
  who never pass args. The `--` separator is optional sugar.
- **Educational.** A new example demo (`demos/classify.mlpl`)
  uses every new construct so a reader can copy-paste.

## Non-goals

- **Type system.** The strings -> number conversion uses the existing
  `Value::Result` (saga 29 step 012); no new value tagging. Booleans
  stay 0/1 floats (audit #3 is a separate saga).
- **Variadic CLI flags.** `args()` returns a `Value::StrList`; the
  script is responsible for its own flag parsing. No `getopts`,
  no `clap`-style declarative parsing.
- **Async I/O.** `read_stdin()` blocks. Multi-stream piping is out.
- **REPL inline `if`/`while`.** The constructs work at top level
  in script mode; making them feel natural at the `mlpl>` prompt
  (multi-line entry, etc.) is a UX polish step that can wait.

## Dependencies

- **`Value::Result`** (saga 29 step 012). `env(name)`, `to_number(s)`,
  and stdin reads return Result to surface failure modes.
- **`Value::StrList`** (saga 29 step 002). `args()` returns one.

## What already exists

- `mlpl-repl -f script.mlpl` runs a script and prints the final
  expression's value (`apps/mlpl-repl/src/main.rs`).
- `Value::Result` + accessors: `is_ok`, `is_err`, `unwrap`,
  `err_message`, `unwrap_or` (docs/lang-reference.md:401-405).
- `Value::StrList` (string list) construction via `["a", "b"]`
  literals (saga 29 step 002).
- `repeat N { body }` and `train N { body }` loops with body
  values captured to `last_losses` / `last_rows`. No `break` or
  `continue`.

## Quality requirements (every step)

Identical to saga 30. TDD; cargo test + clippy + fmt +
markdown-checker + sw-checklist green; `/mw-cp` checkpoint; push
after every commit. Each new builtin ships with at least one
focused test; each new parser construct ships with parser tests +
eval tests.

## Steps

### Step 001 -- print + eprint builtins

Warmup. `print(value)` writes the value's display form (the same
format the REPL uses for terminal output) to stdout, followed by
a newline. `eprint(value)` writes to stderr. Both return their
argument unchanged so they compose: `x = print(some_computation)`
binds and shows.

TDD: unit tests in mlpl-runtime that capture stdout / stderr (or
use a writer-injecting test harness) and assert the rendered
text matches the display contract. Touch
`crates/mlpl-runtime/src/builtins.rs`, register the names, route
to a small writer-aware impl.

### Step 002 -- to_number + to_int + env builtins

Three builtins that all return Result. `to_number("42")` returns
`Ok(42.0)`. `to_int("3.5")` returns `Err("not an integer")`.
`env("MODEL_PATH")` returns `Ok("/path")` if set, `Err("MODEL_PATH not set")`
otherwise. Group together because they share the Result-returning
shape and the migration cost is identical.

TDD: unit tests that exercise each builtin's happy path and each
specific error class.

### Step 003 -- args() builtin + CLI passthrough

Two parts:

1. `args()` builtin returns a `Value::StrList` of the trailing CLI
   args. Empty list when run via the REPL with no script. Lives in
   `crates/mlpl-runtime/src/builtins.rs`.
2. `mlpl-repl` accepts trailing positional args after `-f
   script.mlpl`. Use the `--` separator convention:
   `mlpl-repl -f s.mlpl -- foo bar`. Without `--` the existing
   behavior is preserved (no args).

TDD: integration test in `crates/mlpl-eval/tests/` that constructs
a session with pre-set args and asserts `args()` returns the right
StrList; a separate REPL test in `apps/mlpl-repl/` that spawns
the binary with `-f ... -- arg1 arg2` and asserts the script's
output reflects those args.

### Step 004 -- if cond { then } else { else } expression

Parser change: add `Expr::If { cond, then, else_ }` to the AST.
Surface form: `if cond { then_expr } else { else_expr }`. The
`else` clause is required (no dangling-if). `cond` is truthy iff
non-zero (matches existing convention for `Value::Number` and
`Value::Result.ok`).

TDD: parser tests on the surface form; eval tests on `if 1 { 42 }
else { 99 } == 42` and `if 0 { 42 } else { 99 } == 99`; eval tests
with non-scalar `cond` that the type system rejects with a clear
error message.

### Step 005 -- while + break + continue

Parser change: add `Expr::While { cond, body }` to the AST. Add
`Expr::Break(Option<Expr>)` and `Expr::Continue` as control-flow
expressions. `break value` from inside any loop (`while`, `repeat`,
`train`, `for`) makes the loop return `value`. `continue` skips to
the next iteration. `break` outside a loop is a parse error.

TDD: parser tests; eval tests for each control flow path; a
test that exercises `break` inside `repeat N { ... }` to confirm
back-compat with the existing loops.

### Step 006 -- stdin + script exit code + Err propagation

Three small additions in the same step:

1. `read_stdin()` blocks until EOF and returns `Value::Str` with the
   contents. `read_stdin_lines()` returns a `Value::StrList`.
2. `exit(code)` builtin terminates the script with the given integer.
3. In `mlpl-repl`'s `-f` mode, if the script's final expression
   evaluates to `Err(msg)`, exit non-zero and print `msg` to stderr.

TDD: harness that pipes mock stdin into the REPL binary; assertion
that an `Err(...)`-tailed script exits with code 1 and prints the
message; assertion that `exit(0)` ends cleanly and `exit(2)` ends
with code 2.

### Step 007 -- example script demo + usage doc

`demos/classify.mlpl`: a script that takes a path argument via
`args()`, loads the image with `load_images`, picks a model
based on an `--model` flag (parsed by hand from the StrList),
runs inference, prints the label + confidence, and exits non-zero
on an unreadable image. Touches none of the runtime; pure
composition of step 001-006 surface.

Also add a "Scripting in MLPL" section to `docs/usage.md` walking
the four critical constructs with the demo as the worked example.

TDD: integration test in `crates/mlpl-eval/tests/` that
exercises the demo end-to-end with a known image path and
asserts the printed-label match.

### Step 008 -- close out audit findings

Mark #22, #23, #24, #25, #26, #27, #28, #29, and #30 as SHIPPED
in `docs/language-audit.md` with commit SHAs; move them into the
Shipped subsection of `docs/plan.md`'s Breaking-change candidates;
refresh CHANGES.md; update docs/language-status.md.
