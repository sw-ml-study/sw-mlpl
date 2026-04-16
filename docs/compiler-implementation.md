# Compiler Implementation (Educational Tour)

This doc walks through *how* the MLPL compiler is built, phase by
phase, with pointers to the actual source. Intended reading for
someone who wants to learn compiler construction from a real
(small) language, or for a contributor ramping up on the
codebase.

The companion docs are:

- [`docs/compiler-guide.md`](compiler-guide.md) -- how to *use*
  the compiler (`mlpl!` macro, `mlpl build`)
- [`docs/compiling-mlpl.md`](compiling-mlpl.md) -- why the
  compile path exists at all
- [`docs/architecture.md`](architecture.md) -- crate-level
  dependency flow

All line counts below reflect the `v0.10`-era tree. Small numbers
mean you can read every file in an afternoon -- that's on
purpose.

## 1 - The big picture

MLPL has **two** execution paths sharing the same front end:

```text
source.mlpl
    |
    v
+---------+    tokens     +----------+    AST      +-----------+    result
|  lexer  |-------------->|  parser  |------------>| evaluator |---------> stdout / REPL
+---------+               +----------+             +-----------+
  (mlpl-parser::lexer)      (mlpl-parser::parser)    (mlpl-eval)

                                        +-- lower --+    Rust tokens   +--------+    binary
                                        |           |----------------->| rustc  |---------> ELF/Mach-O
                                        +-----------+                  +--------+
                                        (mlpl-lower-rs)                 (via cargo)
```

Front end = **lex + parse**, produces an `Expr` AST. Back ends =
either **evaluator** (interpreter, `mlpl-eval`) or **lowerer**
(`mlpl-lower-rs` emits a Rust `TokenStream`, handed to rustc via
the `mlpl!` proc macro or the `mlpl build` CLI).

There is deliberately **no third back end**. No LLVM, no JIT, no
bytecode VM. The language's closed surface (no `exec(string)`,
no runtime code loading, see `docs/compiling-mlpl.md` "Why this
works") lets AOT-to-Rust be the whole compile story.

## 2 - Lexing (`crates/mlpl-parser/src/lexer.rs`, ~180 lines)

**Approach: hand-rolled state machine over `&str` bytes.**

No lexer generator (no logos, no lalrpop-lexer). Why:

- The token set is small (~25 kinds in `token.rs`) and stable
  enough that a hand-rolled lexer fits on one screen.
- Single-source-of-truth Rust code is easier to trace at a
  breakpoint than a generated table.
- `#[derive(Debug)]` on `TokenKind` means every lexer bug can be
  surfaced with a one-line `dbg!`.

The lexer walks `source.chars()` (UTF-8-aware since the Saga 12
fix, see commit log) and emits a `Vec<Token>`. Each token
carries a `Span { start, end }` of byte indices into the source;
that's threaded all the way through parsing and used for error
reporting with underline carats.

Interesting lexing choices:

- **Newlines are tokens**, not whitespace. They act as statement
  separators alongside `;`, and the parser decides which one to
  consume. This keeps REPL-style programs (one statement per
  line, no trailing `;`) readable without an off-side rule.
- **`#` to end-of-line is a comment**, eaten by the lexer itself.
- **Numeric literals split int vs float by presence of `.`.** The
  parser later promotes ints to f64 when they're used in
  arithmetic with floats; the AST preserves the distinction for
  diagnostics.
- **String literals are UTF-8.** Escapes (`\n`, `\t`, `\\`, `\"`,
  `\'`) are processed in the lexer; the token carries the
  processed `String`, not the quoted source.
- **Keywords (`repeat`, `train`, `for`, `in`, `experiment`) are
  identifiers the lexer promotes to dedicated `TokenKind`
  variants** before returning. This moves keyword dispatch out
  of the parser and keeps `Ident` a free-form bag.
- **`:` is a token** because annotation syntax
  (`x : [batch, feat] = ...`) needs it, not because it's a REPL
  command separator. That's why `:wsid` fails at parse time with
  "unexpected token" -- the lexer produces `Colon, Ident("wsid")`
  and the parser has no expression rule for "bare `:` at the
  start of a statement".

The `lex_util.rs` module carries small helpers (is-digit,
is-ident-start, etc.) and stays boring on purpose.

## 3 - Parsing (`crates/mlpl-parser/src/parser.rs`, ~290 lines)

**Approach: recursive descent + precedence climbing.**

No parser generator (no lalrpop, no pest, no nom). Why:

- The grammar is small (statements, expressions, a handful of
  keyword constructs). The whole parser fits in ~300 lines.
- Precedence climbing handles all the arithmetic without a
  grammar of ~20 recursive rules.
- Error messages can be bespoke. "unexpected token: `Colon` at
  0..1" points directly at the bad byte range, because the
  parser holds the current token's span and emits it verbatim.

Entry point: `fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError>`.
Returns a **vector of statements**, one per top-level expression
or assignment. Everything above statement granularity is handled
at higher layers (the REPL loop consumes one statement at a time
for streaming evaluation).

The `Parser<'a>` struct is trivial: a token slice + a cursor.
Every parse function returns `Result<Expr, ParseError>` and
advances the cursor on success.

Key parsing patterns:

- **Statement dispatch.** `parse_statement` peeks the head token
  and picks a rule: `repeat` / `train` / `for` / `experiment`
  keywords each have their own rule; identifier followed by `=`
  is an assignment; identifier followed by `:` is an annotated
  assignment; anything else is an expression.
- **Precedence climbing for expressions.** `parse_expr(min_prec)`
  parses an atom, then loops: while the next token is a binary
  operator of precedence `>= min_prec`, recurse with
  `min_prec = op_prec + 1`. Handles `*`/`/` tighter than `+`/`-`
  in ~15 lines.
- **Atoms cover literals, parenthesized expressions, function
  calls, array literals, `param[shape]` / `tensor[shape]`
  constructors.** Function calls are just `Ident` followed by
  `LParen` -- the parser doesn't special-case any builtin names;
  every builtin is an ordinary `FnCall` in the AST, resolved at
  eval time or lower time.
- **Annotated assignment** (`x : [batch, feat] = ...`) is
  desugared into an `Assign` whose value is wrapped in a
  synthetic `label(...)` call. Every downstream layer (evaluator,
  lowerer) gets label metadata for free without a new AST node.

Block-scoped constructs (`repeat N { body }`, `train N { body }`,
`for row in X { body }`, `experiment "name" { body }`) all share
the same body-parsing helper, which eats statements until it hits
the closing `}`.

## 4 - AST (`crates/mlpl-parser/src/ast.rs`, ~160 lines)

One enum, many variants. Each variant owns a `Span` so errors
can point at the offending source range, and each variant is the
direct tree node the downstream stages match on.

```rust
pub enum Expr {
    IntLit(i64, Span),
    FloatLit(f64, Span),
    Ident(String, Span),
    StrLit(String, Span),
    ArrayLit(Vec<Expr>, Span),
    BinOp { op: BinOpKind, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
    UnaryNeg(Box<Expr>, Span),
    FnCall { name: String, args: Vec<Expr>, span: Span },
    Assign { name: String, value: Box<Expr>, span: Span },
    Repeat { count: Box<Expr>, body: Vec<Expr>, is_train: bool, span: Span },
    For   { var: String, iter: Box<Expr>, body: Vec<Expr>, span: Span },
    Experiment { name: String, body: Vec<Expr>, span: Span },
    TensorCtor { kind: TensorCtorKind, shape: Vec<Expr>, span: Span },
    // ...
}
```

No visitor pattern, no generic trait infrastructure -- each
back end writes its own `match expr { ... }`. That makes the
tree easy to understand and keeps the per-variant handling
visible at a glance.

## 5 - Interpreter (`crates/mlpl-eval/`)

**Approach: tree-walking evaluator with a mutable `Environment`.**

Entry point: `eval_program(stmts: &[Expr], env: &mut Environment)
-> Result<DenseArray, EvalError>`. Iterates statements, calls
`eval_expr` on each, returns the last statement's value. That's
why `1; 2; 3` at the REPL returns `3`.

The `Environment` holds:

- `vars: HashMap<String, DenseArray>` -- all bindings, keyed by
  name. Params live here too, with a membership check against a
  `params: HashSet<String>` for trainable-tagging in `:vars`.
- `models: HashMap<String, ModelSpec>` -- Saga 11 Model DSL values.
- `tokenizers: HashMap<String, TokenizerSpec>` -- Saga 12.
- `optim_state: OptimizerState` -- per-parameter `m` / `v` buffers
  keyed by param name; survives across `adam` calls in the same
  `train` loop.
- `experiment_log: Vec<ExperimentRecord>` -- Saga 12.
- `data_dir` / `exp_dir` / `rng` / ... -- I/O knobs the terminal
  REPL sets from CLI flags.

`eval.rs` is the main dispatch. Big `match *expr` over `Expr`
variants, each arm delegating to:

- `eval_ops.rs` for binary/unary arithmetic (broadcasting is in
  `mlpl-array`).
- `eval_for.rs` for `for row in X { body }` (separate module
  because the streaming-iter story is bigger than a single
  match arm).
- `grad.rs` for `grad(expr, wrt)` -- lifts the expression onto
  `mlpl-autograd`'s reverse-mode tape, backprops, returns the
  gradient.
- `model_dispatch.rs` for `apply(model, X)` and `:models`
  introspection. `model_tape.rs` for the tape-lowered versions
  used inside `grad(...)`.
- `loader.rs` for `load` / `load_preloaded` (terminal I/O +
  compiled-in corpus registry).
- `bpe.rs` for the BPE trainer and `tokenizer.rs` for
  encode/decode.
- `experiment.rs` for the `experiment { ... }` scoped form.

No IR, no bytecode. Tree walking with `Box<Expr>` indirection is
O(depth) in call stack but that's fine for tutorial-scale
programs; the Saga 13 Tiny LM trains in seconds. The
`cargo bench -p mlpl-bench` numbers in `docs/benchmarks.md` are
the real comparison against the compile path.

## 6 - Autograd tape (`crates/mlpl-autograd/`)

**Approach: reverse-mode autodiff via an explicit tape**, not
operator overloading on a graph.

`grad(expr, wrt)` at the language level calls into
`mlpl-autograd`. The tape is an append-only `Vec<TapeEntry>`;
each entry records an op (add / mul / matmul / softmax / ...),
the input tensor indices, and a local Jacobian or a closure that
computes one on demand.

Forward pass: evaluate the expression against the tape,
allocating a fresh tape entry per op and returning a `Tensor`
handle (index into the tape).

Backward pass: seed the output's adjoint with `1.0`, walk the
tape in reverse, at each entry add the op's contribution to its
inputs' adjoints (chain rule). Reads out `tape.adjoint(wrt)` as
a `DenseArray` with the same shape as `wrt`.

Why a tape and not overloaded operators: the tape is explicitly
reset per `grad()` call, so there's no hidden state to manage.
Testing is easier too -- `gradcheck` in `backward.rs` compares
backprop against finite-differences on every supported op.

`mlpl-autograd` is interpreter-only. The compile path doesn't
lower `grad` because the tape is runtime state; lowering it
would require emitting the tape-building code at Rust compile
time, which is a separate saga.

## 7 - Lowerer (`crates/mlpl-lower-rs/`, ~380 lines total)

**Approach: AST walk that emits a `proc_macro2::TokenStream`
using `quote!`**. Each `Expr` variant lowers to a Rust
expression of type `::mlpl_rt::DenseArray`.

The whole lowerer is `match expr { ... }` over `Expr` variants.
Interesting pieces:

- **Scalar literals wrap via `DenseArray::from_scalar`**. Arrays
  are uniformly rank-0+. No separate `f64` code path.
- **Binary ops go through `DenseArray::apply_binop(op, &rhs)`**.
  The operator is an enum selected by the lowered match arm.
- **Variable binding uses Rust `let`**. `Assign { name, value }`
  becomes `let name = <lowered value>;` in the output token
  stream. `Ident` reads become `name.clone()` because
  `DenseArray` isn't `Copy` and clippy doesn't like implicit
  moves across loop iterations.
- **Function calls live in `fncall.rs`**. One big `match name`
  inside `lower_fncall`, mapping each supported builtin to the
  matching `::mlpl_rt::...` path. Unknown names return
  `LowerError::Unsupported`.
- **Static label check for matmul.** When both operands' labels
  are statically known at lower time (traceable via `labels_of`
  in `fncall.rs`), the lowerer verifies the contraction axes
  agree. Mismatch returns `LowerError::StaticShapeMismatch`
  instead of emitting broken Rust. The proc macro converts this
  to a `compile_error!` with span carats.
- **Annotations lower transparently.** Because the parser
  desugars `x : [batch, feat] = ...` to
  `Assign { value: FnCall { name: "label", ... } }`, the lowerer
  just handles `label(...)` like any other function; labels flow
  through without a dedicated AST node.

The output is straight Rust. No intermediate IR, no SSA, no
optimization pass -- rustc sees the token stream and does its
own lowering to LLVM IR from there. That's the design-level
reason the compile path is small: every one of those harder
layers already exists in rustc, and MLPL lets the token stream
be the handoff point.

What's deliberately deferred (returns `LowerError::Unsupported`
today): `repeat`, `train`, `for`, `TensorCtor` (param / tensor),
`grad`, `adam`, the whole Model DSL, string-named axis reductions,
and every Saga 12+ surface. See `docs/compiler-guide.md` "What the
compile path lowers today" for the full breakdown. Extending any
of these means writing the lowering rules *and* keeping runtime
semantics (`mlpl-rt`) in sync.

## 8 - Runtime target (`crates/mlpl-rt/`)

Compiled MLPL programs link only against this crate (plus
`mlpl-array` and `mlpl-core` transitively). It re-exports
`DenseArray`, `Shape`, `LabeledShape`, and provides typed Rust
signatures for every primitive the lowerer emits:
`mlpl_rt::iota(n)`, `mlpl_rt::reshape(&a, &[2, 3])`,
`mlpl_rt::matmul(&a, &b)`, etc.

Deliberately does *not* depend on `mlpl-eval`, `mlpl-runtime`, or
`mlpl-parser`. Those are tooling. The compiled binary doesn't
carry an interpreter.

## 9 - Proc macro + CLI on top of the lowerer

`crates/mlpl-macro/src/lib.rs` is the `mlpl!` proc macro. Takes
the token stream fed by rustc, re-lexes it as MLPL source, parses
with `mlpl-parser`, lowers with `mlpl-lower-rs`, and returns the
resulting Rust `TokenStream` back to rustc. Errors become
`syn::Error` -> `compile_error!` with span carats.

`apps/mlpl-build/src/main.rs` is the `mlpl build` subcommand.
Runs the same lex-parse-lower pipeline eagerly (so syntax /
static-label errors surface with `mlpl-build:` prefixed stderr
*before* rustc is invoked), generates a one-file cargo project
whose `main.rs` wraps the source in `mlpl! { ... }`, invokes
`cargo build --release`, and copies the resulting binary to the
requested output. `--target <triple>` is forwarded to cargo for
cross-compilation.

## 10 - Trace exporter (`crates/mlpl-trace/`)

Interpreter-side only. When `:trace on` is active, `mlpl-eval`
populates a `Trace` with per-op events (op name, input shapes,
output shape, span). `:trace json` pretty-prints the trace as
JSON. Useful for debugging and for structural analysis in other
tools; the schema round-trips through serde.

Not in the compile path by design -- tracing is a REPL feature,
and a compiled binary shouldn't pay the overhead.

## 11 - What's *not* here (on purpose)

- **No grammar DSL.** Hand-written lexer + recursive descent.
  Saves a build dep and keeps the grammar readable by anyone who
  can read Rust.
- **No IR.** AST in, Rust `TokenStream` out. rustc is the IR.
- **No LLVM crate.** `llvm-sys` / `inkwell` would be a
  significantly bigger ask for a build, and Rust already has a
  perfectly good LLVM integration via rustc itself.
- **No JIT.** `cranelift` would fit, but the "compiled binary
  ships without an interpreter" property is the whole point;
  JIT would drag a JIT engine into the binary.
- **No `exec(string)`.** Intentionally absent. See
  `docs/compiling-mlpl.md` "Why this works".
- **No bytecode VM.** The interpreter is tree-walking; if it
  needs to be faster, the Saga 14 MLX backend is the intended
  path, not a bytecode detour.
- **No macro system in the language itself.** MLPL's macro story
  is Rust's -- `mlpl!` embeds MLPL in Rust, not the other way
  around.

## 12 - Reading order for a new contributor

1. [`crates/mlpl-parser/src/token.rs`](../crates/mlpl-parser/src/token.rs)
   -- see every token MLPL has (~65 lines)
2. [`crates/mlpl-parser/src/ast.rs`](../crates/mlpl-parser/src/ast.rs)
   -- see every AST node (~160 lines)
3. [`crates/mlpl-parser/src/lexer.rs`](../crates/mlpl-parser/src/lexer.rs)
   -- hand-rolled lexer (~180 lines)
4. [`crates/mlpl-parser/src/parser.rs`](../crates/mlpl-parser/src/parser.rs)
   -- recursive descent + precedence (~290 lines)
5. [`crates/mlpl-eval/src/eval.rs`](../crates/mlpl-eval/src/eval.rs)
   -- tree-walking interpreter main dispatch (~500 lines)
6. [`crates/mlpl-lower-rs/src/lib.rs`](../crates/mlpl-lower-rs/src/lib.rs)
   -- AST-to-Rust lowerer (~230 lines)
7. [`crates/mlpl-rt/src/lib.rs`](../crates/mlpl-rt/src/lib.rs)
   -- the runtime the compiled code calls into
8. [`crates/mlpl-autograd/src/tape.rs`](../crates/mlpl-autograd/src/tape.rs)
   -- reverse-mode autograd tape (~145 lines)
9. [`crates/mlpl-macro/src/lib.rs`](../crates/mlpl-macro/src/lib.rs)
   -- the `mlpl!` proc macro (thin wrapper)
10. [`apps/mlpl-build/src/main.rs`](../apps/mlpl-build/src/main.rs)
    -- `mlpl build` subcommand (~170 lines)

Total: under 2000 lines across the compile path. Every file is
small enough to read top to bottom; the value is in the shape
of the pipeline, not in clever tricks inside any one phase.

## 13 - Why build it this way

MLPL targets two readers:

1. **People learning array programming.** A small,
   hand-written compiler makes the language feel
   approachable. "I could write this" is a valid response;
   the `docs/compiler-implementation.md` you're reading is
   a first step toward that.
2. **AI coding agents using MLPL as a substrate.** A closed
   language surface with statically knowable shapes and
   explicit compile errors is friendlier to agents than a
   dynamic language with `exec` escapes, and the compile path
   gives agents a deployment target.

Both audiences benefit from the same choice: the compiler is
boring, readable, and small. No magic.
