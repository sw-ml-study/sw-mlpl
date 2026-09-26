# Downstream updates: what to adopt, what to delete

A planning document for the repos that USE sw-mlpl (the `demo-*`
repos, moe-microscope, reasoning-from-scratch, demo-mlpl-libraries,
mlplunit, demo-extensions, demo-decision-model, microgpt-mlpl, ...).
It lists every sw-mlpl fix and feature that retires a documented
downstream workaround, the behavior changes to re-check, the planned
work that will retire more, and the style sw-mlpl recommends for new
code. Evidence comes from each repo's own request / issue ledger and a
pattern count over 1,658 downstream `.mlpl` files (2026-09-23).

How to adopt an item:

1. Re-run the repo's capability probe for it (most repos have one:
   `just capabilities`, `scripts/run-probes`, `tests/test_capability_*`).
   A fix shows up as DRIFT on the named probe.
2. Flip the probe expectation, delete the workaround, and re-pin the
   sw-mlpl commit in the repo's ledger.
3. Run the repo's gate (parity / tangle / regression) so drift across
   scripts, `lib/` and literate docs is caught.

All items below are on `main`; the adjacent checkout's
`target/release/mlpl-repl` is rebuilt at every push.

## 1. Shipped: delete these workarounds

| Change (commit) | Retires | Repos (ledger id) |
|---|---|---|
| `unpack(bytes, dtype)` -- bulk decode of a byte buffer to a flat array in one native pass, every `reinterpret` dtype incl. `bf16` / `f16` (b3180d9a) | one-value-at-a-time `read_bf16_le` loops; the planned native `sten:read_tensor` fallback | reasoning-from-scratch (R11, was blocking the weight loader); demo-extensions (E3 decision) |
| `is_result(x)` -- total predicate, never raises (b3180d9a) | `str_eq(type_of(r), "record")` classification of extension results | demo-extensions (R1); demo-mlpl-libraries and reasoning-from-scratch (`type_of` + `str_eq` lines: 30 and 18) |
| `get_error` on a string error names `err_message(r)` (b3180d9a) | nothing to delete: `err_message(r)` IS the supported way to read a string error | demo-extensions (R2) |
| `u:` function parameters work as `cross_entropy` targets, `rotate` shifts, `pow` exponents, `transpose_axes` permutations inside `grad` (67c2ca86) | global-variable losses with no arguments (`global_set` of tokens / targets / mask) | microgpt-mlpl (a) |
| comparison operators (`<` `>` `<=` `>=` `==` `!=`) inside `grad` are constant 0/1 masks, like `lt` / `gt` / `eq` (67c2ca86) | building masks outside the loss | microgpt-mlpl (b) |
| multi-head `attention` / `causal_attention` documented as differentiable for any `heads` (67c2ca86; was a stale doc line) | hand-written multi-head attention chosen because of the doc | microgpt-mlpl (c) |
| string-valued statements (`print("...")`, `name = "..."`) inside `repeat` / `train` / `for` bodies (e6ee2203) | `while` loops used only to dodge the error | microgpt-mlpl (j) |
| `shape` / `rank` / `len` of a function's own argument inside `grad` (a6139122) | passing sizes in as extra arguments | re-probe moe-microscope F23 (reshape dims from `shape()`) |
| grad never silently constant-folds a `u:` call whose body reads a param; unsupported forms are named in the error (ab858695) | nothing to delete -- but see section 2 | demo-decision-model (Q5), moe-microscope (F17, F25 symptoms) |
| layer weights API: `params(model)`, `get_param` / `set_param` by role, `rms_norm(d, {eps})` on `[rows, d]` or `[B, T, d]`, bias-free `linear(in, out, seed, {bias: 0})` (8d13e571) | hand-written `u:linear` / `u:rmsnorm` kept only to match an equation or a parameter count; weight loading through generated parameter names | microgpt-mlpl (requests #10), reasoning-from-scratch (`u:qwen_rms_norm`, bias-free Qwen projections) |
| a `u:` call costs O(names it writes), not a copy of every global: 0.557 -> 0.0006 ms per call with a 2.28M-element global in scope, now independent of global size (grad-soundness-records step 009) | `expunge` of big globals before hot loops; restructuring to avoid `u:` calls in inner loops (the per-READ copy of a large global is the separate cow-values saga) | microgpt-mlpl (e, call-cost half) |
| `gather_rows` and `embed` differentiate through a native row gather: O(n x d) forward and backward, independent of the table's row count (20,000 ids into a 50,000-row table: ~10 ms; the old one-hot form needed an 8 GB `[n, V]` matrix) (grad-soundness-records step 010) | mini-batching or vocabulary truncation adopted only because full-batch gathers took ~30 s per step | demo-decision-model (full-batch scorer training), microgpt-mlpl, reasoning-from-scratch embeddings |
| record field reads inside `grad` are constant leaves, and a record may be passed to a `u:` function called inside the loss (grad-soundness-records step 011) | differentiated entry points that take every field as a separate array argument; binding fields to variables before a traced loss | demo-decision-model (Q5: the 12-argument `lib/` entry points can take one `{ids, wmask, ...}` record), moe-microscope (F17) |

## 2. Behavior changes to re-check

These are corrections, but a script that relied on the old behavior
changes result:

- **An optimizer step errors when a listed, non-frozen param gets no
  gradient** (ab858695). `adam` / `momentum_sgd` used to zero-fill it
  and silently skip the update -- the symptom of moe-microscope F25
  (`if` inside `grad` detaching the tape) and of a folded
  subexpression. Drop the param from the list or `freeze()` its model.
  An `if` inside a traced loss now errors by name instead of training
  nothing.
- **`grad` of a `u:` call whose body reads a param through an
  unsupported builtin now errors** instead of returning a gradient
  missing that term (ab858695). If a probe pinned the old wrong
  gradient, it was pinning the bug.
- **Comparison operators inside `grad` no longer error** (67c2ca86).
  A negative probe asserting the old error will flip.
- **`get_value` / `get_error` error text changed** (b3180d9a): it now
  names `unwrap(r)` / `err_message(r)`; string-matching tests need
  the new wording.

## 3. In flight and planned: workarounds these will retire

The grad-soundness-records saga's items are all in section 1. Next is
the Python-ML-developer ergonomics program
(docs/future-sagas-queue.md), each with the pattern it deletes and
the heaviest users (line counts over downstream `.mlpl`):

| Saga | Adds | Retires | Heaviest users |
|---|---|---|---|
| readable-scripts | `format(...)` (Python format specs), `write(s)`, variadic `str_concat`, `and` / `or` / `not`, record destructuring | `str_concat(str_concat(` towers; `* (1 - done)` booleans; field-by-field unpacking; hand-rolled `{:4d}` formatters | moe-microscope 226, demo-decision-model 44, demo-abstract-algebra 40 nested concats |
| diagnostics | `file:line:col` for the failing inner statement; mlpl-mode keywords; `include` in `--babel-session` | bisecting silent-position errors; duplicated `lib/` in literate docs | all |
| tensor-indexing | differentiable `gather(x, idx[, axis])`, `slice(x, lo, hi[, axis])`, multi-axis `at(x, i, j)` | `reshape(gather_rows(...))` single-index gathers (`u:gather1`), `take(take(` | demo-abstract-algebra 150 / 147, moe-microscope 62, microgpt-mlpl 60 |
| param-groups | `param_init({...}, seed, std)` returning a name group `adam` / `grad` / `params` accept | inline 9-name `adam` lists; 18-line param declarations | microgpt-mlpl (f) |
| lists-and-text | `list_at`, `for s in string_list`, `list_append` / `list_concat`, `codepoints` | `unwrap(list_get(`; `;`-join + `str_split` round trips; ASCII-only assumptions | moe-microscope 187, demo-abstract-algebra 56, demo-coding-agent 37 |
| cow-values | shared copy-on-write arrays and records | pre-encoding to dodge read copies; trace-instead-of-mutate; quadratic append loops | microgpt-mlpl (e), reasoning-from-scratch (R10), demo-funtional-pipelines |

## 4. Declined or redirected (keep your version)

- `x[i]` subscripts -- functions only (`at`, `gather`, `slice`).
- Records of param VALUES in `adam` -- records hold copies, so they go
  stale after an update; use a name group (param-groups saga) or a
  model.
- `where(x, cond)` meaning compress -- `where` will only ever mean
  NumPy's `np.where(cond, a, b)`; filter with `compress(mask, x)`.
- `select(xs, :u:p)`, `char_class`, `str_trim`, `str_replace` --
  library functions (demo-mlpl-libraries `text`).
- Closures, lambdas, eval-string, full macros.
- Symmetric-Result extension calls -- would break every consumer;
  `is_result` is the supported branch test.
- Byte-matching another implementation's RNG / sampling rule --
  parity code belongs in the parity repo (e.g. `lib/splitmix64`).

## 5. Reported and not yet triaged

Tracked here so the next triage pass does not lose them (source: each
repo's ledger):

- reasoning-from-scratch R3 (batched rank-3 `matmul`; attention slices
  per head meanwhile), R10 (element access cost scales with container
  size -- overlaps cow-values).
- moe-microscope F7 (a model as a `u:` argument), F8 (`emit_frame`
  needs a literal name), F23 / F25 (re-probe after the fixes above).
- demo-extensions: applet fs-root config, terminal-error supervision,
  "derived scalar reuse" shape mismatch, extension handle as a `u:`
  parameter.
- demo-decision-model Q1 (`freeze` as an expression), Q3 (`experiment`
  with a computed name), Q4 (optimizer state per model, not global).
- microgpt-mlpl g (mlplbench sandbox root), i (document that `adam`
  returns the pre-update loss), k (`parse_json` nested arrays).
- demo-abstract-algebra A2 (deep recursion aborts), A3, A4, C1-C3,
  D1-D3, E1-E4, #24, #25.
- demo-coding-agent F6 (`include` resolution differs repl vs
  mlplunit), F7 (`else if` / `match`), F9 (stdin TTY).
- demo-linear-algebra B1 (batched matmul, same as R3), B2 (stable
  solve), B3 (ordered SVD).
- demo-funtional-pipelines: `reduce(:u:step, ...)` over a user
  callable.
- demo-category-theory: function-ref lists, callable-returning
  compose (Track 8 `apl2-hof-and-order`).
- mlplunit roadmap: modules / imports, `assert_raises`, process
  controls.
- demo-memory 3 (packed layout with observable size), 4 (passable
  seeded RNG).

## 6. Writing idiomatic MLPL

The house style for new downstream code. Prefer the built-in over an
equivalent written locally, and when a local version must exist (for
parity, or pending an upstream fix) mark it with the ledger id it
waits on, so the next person deletes it on schedule.

**Arrays, not loops.**

- Whole-array expressions first; a `while` over indices is the last
  resort and deserves a comment saying why.
- Masks are arrays: build them with comparisons and use them with
  `compress(mask, x)` (filter), `mask * x` (select) or
  `reduce_add(mask)` (count).
- Outer products come from broadcasting a column against a row
  (`reshape(r, [k, 1]) * reshape(r, [1, k])`) or `table(:u:f, a, b)`;
  see `examples/primes.mlpl` for the classic sieve.
- Per-element logic is `each(:u:f, v)`, not an index loop.
- Build strings with `str_join(parts, sep)` (linear time), not
  repeated `str_concat` (quadratic).

**Functions, values, results.**

- Pass data in as arguments -- inside `grad` too, where function
  parameters now resolve everywhere. Globals are for configuration
  and parameters, not for threading data into a loss.
- Every `def` opens with a docstring; run `scripts/mlpl-fmt.sh`.
- A fallible operation returns a Result: chain with postfix `?`,
  default with `unwrap_or`, read a string error with `err_message`,
  and branch on a maybe-Result with `is_result`.
- Records are data. Destructure them outside a traced loss and pass
  the fields as arrays.

**Models and training.**

- Prefer the Model DSL (`linear`, `embed`, `rms_norm`,
  `causal_attention`, `chain`, `residual`) and `adam(loss, model,
  ...)`; the parameter list comes from the model.
- To load pretrained or reference weights, `set_param(model, role,
  value)`; never write the generated `__attn_Wq_0`-style names.
- Match a paper's equation with layer options (`{bias: 0}`,
  `{eps: ...}`) rather than a hand-written layer.
- Use `train N { ... }` and read `last_losses`; `adam` returns the
  step's pre-update loss, so no extra forward pass is needed to log.
- Load weight tensors with bounded `read_bytes_packed(p, off, n)` +
  `unpack`, tensor by tensor (MLPL arrays are f64: 8 bytes per
  value).

**Where a feature belongs.** Core only for what a library or
extension cannot provide (autograd rules, primitives, dtypes, syntax,
performance); text utilities, formats and protocols are library or
extension (reasoning-from-scratch `docs/feature-homes.md`).

**Further reading.** microgpt-mlpl `docs/idiomatic-mlpl.md` (a worked
catalog of idioms and anti-patterns),
demo-category-theory `AGENTS.md` and `docs/sw-mlpl-capabilities.md`,
demo-mlpl-libraries `docs/library-contract.md`, demo-ml-utils
`docs/math-notation.md`, and sw-mlpl `docs/lang-reference.md`.
