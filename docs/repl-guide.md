# REPL Guide

MLPL ships two REPLs that share an evaluator:

- **`mlpl-repl`** -- terminal, `cargo run -p mlpl-repl`. Has
  filesystem access (sandboxed via `--data-dir`), can write
  experiment runs to disk (`--exp-dir`), and exposes tracing.
- **`mlpl-web`** -- Yew/WASM, deployed at
  <https://sw-ml-study.github.io/sw-mlpl/>. Same evaluator,
  compiled-in corpora for `load_preloaded`, inline SVG rendering,
  a tutorial panel, and a demo selector. No filesystem, no
  tracing UI yet.

Both surfaces accept the same MLPL language and the same
`:`-prefixed introspection commands (with a small terminal-only
extension for `:trace`).

## Starting a session

Terminal REPL:

```bash
cargo run -p mlpl-repl                       # interactive
cargo run -p mlpl-repl -- -f script.mlpl     # run a file, then exit
cargo run -p mlpl-repl -- --data-dir ./data  # sandbox for load()
cargo run -p mlpl-repl -- --exp-dir ./runs   # where experiments write
cargo run -p mlpl-repl -- --trace            # trace enabled from the start
cargo run -p mlpl-repl -- --version
```

Web REPL:

- Open <https://sw-ml-study.github.io/sw-mlpl/>
- The live build is whatever's in `pages/` on `main`
- To iterate locally: `cd apps/mlpl-web && trunk serve --release`

## REPL commands

Every command starts with `:`. Commands are REPL-surface dispatch,
not language syntax -- they don't go through the lexer and can't
appear inside `.mlpl` source files compiled by `mlpl build` (see
`docs/compiler-guide.md`). Scripts loaded via `mlpl-repl -f
script.mlpl` also can't contain them today; the terminal REPL
parses the whole file in one shot, so a `:` line errors with
"unexpected token". Per-line file mode is a future ergonomics
fix.

MLPL's command set is APL-inspired -- `)VARS`, `)FNS`, `)WSID`,
`)CLEAR` map directly. See also the "Workspace Introspection"
tutorial lesson (in the web REPL) for a guided walk.

### Help

| Command | Behavior |
|---|---|
| `:help` | One-screen command + syntax cheatsheet. |
| `:help <topic>` | Focused help for one area. Topics: `vars`, `models`, `fns`, `builtins`, `wsid`, `describe`. |
| `:version` | MLPL version + target architecture. The terminal REPL extends this with git commit SHA and build timestamp. |

### Inspecting the workspace (APL `)WSID`, `)VARS`, ...)

| Command | APL analogue | Behavior |
|---|---|---|
| `:wsid` | `)WSID` | Summary: variable / parameter / model / optimizer-slot counts. |
| `:vars` (or `:variables`) | `)VARS` | List every bound variable with its labeled shape. A `[param]` tag marks trainable leaves. |
| `:models` | -- | List every bound model with its layer structure and parameter count. Specific to MLPL's Model DSL. |
| `:fns` (or `:functions`) | `)FNS` | User-defined function table. Placeholder today -- user functions aren't a language feature yet; points at `:builtins` instead. |
| `:builtins` (or `:built-ins`) | -- | Every built-in, grouped by category, with one-line docs. |
| `:describe <name>` | -- | Rich detail for one name: an array's shape + values preview, a param's shape + trainable tag, a model's layer tree + parameter shapes, a tokenizer's vocab + merge count, or a built-in's signature + one-line doc. |
| `:experiments` | -- | List every recorded `experiment "..." { ... }` run (memory + disk), sorted by timestamp. |
| `:clear` | `)CLEAR` | Reset the workspace: drops all variables, params, models, optimizer state, experiments. Does not re-run any init. |

MLPL does *not* expose APL's `[]IO` index-origin toggle. Array
indexing is 0-origin throughout. `)SAVE` / `)LOAD` and `)SI`
(state indicator / execution stack) don't have analogues yet.

### Tracing (terminal REPL only today)

| Command | Behavior |
|---|---|
| `:trace on` | Enable per-op trace recording for subsequent evaluations. |
| `:trace off` | Disable trace recording. |
| `:trace` | Print a short summary of the last recorded trace. |
| `:trace json` | Print the last trace as JSON to stdout. |
| `:trace json <path>` | Write the last trace as JSON to a file. |

Tracing is wired on the Rust side via `mlpl-trace`; the web REPL
doesn't have a UI for it yet.

### Leaving

| Command | Behavior |
|---|---|
| `exit` (or Ctrl-D) | Quit the terminal REPL. |
| (web REPL) | Just close the tab. `:clear` resets in-tab state. |

## A concrete tour

This is the "Workspace Introspection" demo distilled. Paste one
line at a time:

```mlpl
:version
:wsid                                                  # all zeros

x = 42
v = iota(5)
M : [batch, feat] = reshape(iota(6), [2, 3])
:vars
:describe v

mdl = chain(linear(2, 4, 11), relu_layer(), linear(4, 2, 12))
:models
:describe mdl

tok = train_bpe("abababab", 260, 0)
:describe tok

W = param[3, 2]
:vars                                                  # W is tagged [param]
:wsid                                                  # counts have moved

experiment "workspace_demo" { loss_metric = 0.25; accuracy_metric = 0.94 }
:experiments
compare("workspace_demo", "workspace_demo")           # same run both sides
:describe matmul                                      # built-in doc
```

The demo is bound to "Workspace Introspection" in the web REPL's
demo selector. The matching tutorial lesson is also titled
"Workspace Introspection".

## Labeled axes in the REPL

Annotation syntax on assignment attaches axis names; `labels(x)`
reads them back. Labels propagate and are reported inline by
`:vars` / `:describe`:

```mlpl
Q : [seq, d_k] = randn(17, [6, 4])
K : [seq, d_k] = randn(23, [6, 4])
:vars
# -> Q: [seq=6, d_k=4]
#    K: [seq=6, d_k=4]

scores = matmul(Q, transpose(K)) / sqrt(4)
:describe scores
# -> scores -- array
#      shape: [seq=6, seq=6]
#      values: ...
```

See `docs/lang-reference.md` "Labeled Axes" for the full
built-in table.

## `load` and `load_preloaded`

Two entry points, different scopes:

- **`load_preloaded("name")`** -- compiled-in corpora, works in
  both REPLs. Current registry: `"tiny_corpus"` (a short pangram)
  and `"tiny_shakespeare_snippet"` (~KB of Shakespeare).
- **`load("path.csv")` / `load("path.txt")`** -- terminal REPL
  only, and only under `--data-dir <path>`. Absolute and
  traversing paths are rejected. Web REPL returns an
  `EvalError::Unsupported` with a pointer at `load_preloaded`.

## `experiment "..." { ... }`

Scoped form that records every scalar assigned to a name ending
in `_metric` plus the shapes of any `param` bindings. In the
terminal REPL with `--exp-dir <dir>`, runs also land on disk at
`<dir>/<name>/<timestamp>/run.json`. Either REPL shows them via
`:experiments`; `compare("a", "b")` prints per-metric deltas.

See the "Experiments" tutorial lesson and the `:experiments`
output in `docs/repl-guide.md`. Worth noting:
`std::time::SystemTime::now()` panics on
`wasm32-unknown-unknown`, so the web REPL uses a monotonic
counter for run timestamps -- ordering is preserved, but the
values are not wall-clock.

## Related

- `docs/lang-reference.md` -- language grammar + builtin tables
- `docs/usage.md` -- user guide with worked examples
- `docs/compiler-guide.md` -- how to get MLPL out of the REPL
  and into a binary
- `docs/benchmarks.md` -- how fast is what
- `apps/mlpl-web/src/help.rs` -- the in-REPL `:help` cheatsheet
- `crates/mlpl-eval/src/inspect.rs` -- `:vars`/`:describe`/etc
  implementation, shared between both REPLs
