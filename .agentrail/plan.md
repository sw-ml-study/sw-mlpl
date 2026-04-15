# Tokenizers, Datasets, and Experiment Tracking Milestone (Saga 12, v0.9.0)

## Why this exists

`docs/plan.md` puts Saga 12 squarely between the compile-to-Rust
saga (v0.8.0, just shipped) and Saga 13's Tiny LM end-to-end.
The Tiny LM needs three things the current repo doesn't have:

1. **A way to get text off disk.** Every demo today hand-builds
   its dataset with `blobs()` / `moons()` / `iota()`. A language
   model needs a corpus.
2. **A way to turn text into integer tokens.** Byte-level gets
   you a 256-vocab baseline; byte-level BPE gets you the real
   tokenizer any modern LM uses.
3. **A way to record what a training run did.** Seed, config,
   metrics -- so a re-run is reproducible and runs are
   comparable.

Saga 12 is the last surface-only saga before the Tiny LM
itself. No new kernels, no new backends, no new optimizer. Just
file IO, tokenization, and a small experiment-capture
primitive.

## Non-goals

- Full Hugging Face compatibility. MLPL ships its own BPE
  trainer with a deterministic tie-breaking rule; loading a
  pre-trained HF tokenizer is a later saga if ever.
- Generic dataset formats (parquet, arrow, jsonl). CSV and raw
  text corpora only for now; the Tiny LM doesn't need more.
- Cloud / remote data sources. Local filesystem only.
- Checkpoint / weight serialization. That's orthogonal; Saga 13
  will motivate it.

## Quality requirements (every step)

Identical to Sagas 11 / 11.5 / compile-to-rust:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.

## What already exists

- `mlpl-array::DenseArray` with row-major storage, labeled
  shapes, and the full op surface from Saga 11.5.
- `mlpl-eval::Environment` with vars / params / models / optim
  state / optimizer buffers.
- Parser that accepts string literals (`"foo"`) and multi-line
  programs with both `\n` and `;` statement separators.
- `Value::Str` variant for strings (used today by the SVG
  builtins -- carries diagram type names).
- `mlpl-trace` JSON export for per-step reproducibility of
  individual REPL runs.

None of these touch `std::fs`. This saga adds file IO for the
first time; that is the key new surface.

## Phases

### Phase 1: Datasets (3 steps)

File IO enters MLPL. Every demo today invents its dataset
procedurally; the Tiny LM will need to read a corpus off disk.

- **`load("path.csv")`** returns a `DenseArray` with the CSV's
  numeric columns. First row is treated as header iff it
  contains any non-numeric token; header names become axis-1
  labels so `labels(X) == "col1,col2,..."`. Non-numeric data
  cells error clearly. Raw text files (`.txt`) return a
  `Value::Str` whole-file blob for the tokenizer phase to chew
  on. Absolute paths disallowed at the `mlpl-web` surface;
  allowed under the terminal REPL behind a
  `--data-dir <path>` flag that scopes relative reads to a
  sandbox directory.
- **`shuffle(x, seed)`** returns a row-permutation of a rank>=1
  array. Labels preserved. **`batch(x, size)`** returns a
  rank-(r+1) array of contiguous batches, padding the last if
  needed. **`split(x, train_frac, seed)`** returns a 2-tuple
  (a labeled-2-element array `[[train_rows], [val_rows]]` is
  out; we'll return `split` as two side-effect-free bindings
  via a new `let (a, b) = split(...)` form only if really
  necessary, otherwise one call returns the train chunk and a
  `val_split` twin returns the val chunk; decide in the step).
- **Streaming iteration.** `for row in dataset { body }` binds
  `row` to each rank-(r-1) slice of a rank-r array, runs body,
  and captures per-iteration values into a
  `last_rows` vector in the environment (mirrors `train`'s
  `last_losses`). This is the escape hatch when a dataset is
  too big for a single-array representation; Phase 2's BPE
  trainer uses it.

### Phase 2: Byte-level BPE tokenizer (3 steps)

Two-stage design: bytes first, BPE on top.

- **`tokenize_bytes(str)`** returns a rank-1 array of byte
  indices (0-255). `decode_bytes(tokens)` inverses.
  Deterministic. No training. Gives the Tiny LM a baseline
  tokenizer to sanity-check end-to-end before BPE.
- **`bpe = train_bpe(text_or_tokens, vocab_size, seed)`**
  trains a BPE tokenizer from a `Value::Str` (the Phase 1
  `load` output for raw text) or an already-byte-tokenized
  rank-1 array. Returns a new `Value::Tokenizer` runtime
  value (a sibling to `Value::Model`) holding the merge table
  + vocab. Deterministic tie-breaking: on ties in merge count,
  pick the lexicographically smallest pair by byte order. Seed
  is threaded through for any randomized sub-sampling of the
  training corpus at larger scales; the core algorithm is
  deterministic given the same input.
- **`tokens = apply_tokenizer(bpe, text)`** and
  **`text = decode(bpe, tokens)`** round-trip: any input
  string `s` satisfies `decode(bpe, apply_tokenizer(bpe, s)) == s`
  for every byte sequence the training corpus could have
  produced. A new `:describe` branch prints `bpe -- tokenizer`
  with vocab size and training-corpus byte count.

### Phase 3: Experiment tracking (2 steps)

Reproducibility as a language construct.

- **`experiment "name" { body }`** is a new scoped form that
  (a) captures the seed / RNG state / current config on entry,
  (b) runs `body`, (c) on exit writes a JSON record to
  `.mlpl-exp/<name>/<unix-timestamp>/run.json` containing the
  entry snapshot, the body's source text, and any scalar
  values assigned to names ending in `_metric` during the run
  (so `loss_metric = ...`, `accuracy_metric = ...` land as
  recorded metrics). In the web REPL with no fs, the record is
  held in an environment-scoped list and surfaced via a new
  `:experiments` REPL command; the terminal REPL actually
  writes to disk.
- **`:experiments` / `compare(name_a, name_b)`** REPL
  introspection. `:experiments` lists every recorded run with
  its name, timestamp, and top-line metric. `compare("a", "b")`
  prints a side-by-side of the two runs' metric dicts.

### Phase 4: Docs, tutorial, release (1 step)

- New "Loading Data" and "Tokenizing Text" tutorial lessons in
  the web REPL, immediately after the "Named Axes" lesson.
- New "Experiments" lesson showing a tiny training loop wrapped
  in `experiment "name" { ... }`.
- Update `docs/are-we-driven-yet.md`: move "load.csv", "custom
  tokenizers", "experiment registry" from CONS/PLAN to HAVE.
- Update `docs/saga.md`, `docs/status.md`, `docs/plan.md`.
- Bump workspace version 0.8.0 -> 0.9.0, banners, pages/.
- Tag `v0.9.0-tokenizers`.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | load-csv-txt | 1 | `load("file.csv")` / `load("file.txt")` with header detection + sandbox |
| 002 | dataset-ops | 1 | `shuffle(x, seed)`, `batch(x, size)`, `split(x, frac, seed)` |
| 003 | streaming-iter | 1 | `for row in x { body }` form + `last_rows` capture |
| 004 | bytes-tokenizer | 2 | `tokenize_bytes`, `decode_bytes`; Value::Tokenizer scaffolding |
| 005 | bpe-train | 2 | `train_bpe(text, vocab_size, seed)` with deterministic tie-breaking |
| 006 | bpe-apply-decode | 2 | `apply_tokenizer` + `decode`; round-trip tests |
| 007 | experiment-block | 3 | `experiment "name" { body }` scoped form + metric capture |
| 008 | experiment-registry | 3 | `:experiments` / `compare(a, b)` REPL commands |
| 009 | tokenizers-release-v090 | 4 | docs, banners, tag |

Nine steps. The original prose sketch had a tenth for
per-variable metric capture inside `experiment`; folded into
step 007 since they share all the same plumbing.

## Success criteria

- A tiny corpus can be loaded end-to-end:
  `text = load("data/tiny.txt")` -> `tokens = apply_tokenizer(bpe, text)`
  -> fits the labeled-array pipeline for Saga 13's Tiny LM.
- Round-trip: for every byte string `s`,
  `decode(bpe, apply_tokenizer(bpe, s)) == s`.
- A simple `experiment "baseline" { ... }` block produces a
  `.mlpl-exp/baseline/<ts>/run.json` that lists seeds, source
  text, and any `_metric`-suffixed scalars.
- `:experiments` in the terminal REPL lists at least one run
  after an `experiment` block completes.
- No compile-path regression: Saga 12 additions are interpreter-
  only for now (mlpl-lower-rs doesn't know about `load`,
  `experiment`, or `Value::Tokenizer`). They return
  `LowerError::Unsupported` in the compile path until a future
  saga extends the lowering.
- Every Saga 11.5 demo still runs unchanged.
- All quality gates green; pages deployed; release tagged.

## Risks and open questions

- **File IO in the web REPL.** `std::fs` doesn't exist in
  WASM; a direct `load("foo.csv")` would hang the parser or
  panic. Proposal: web REPL ships a small preloaded-corpus
  registry (e.g. `load_preloaded("tiny_shakespeare")`), and
  the web UI synthesizes a `Value::Str` from an embedded
  dataset. Terminal REPL reads real files. Decide final API
  in step 001.
- **Sandbox story.** The terminal REPL must not let a random
  `load("/etc/passwd")` succeed. Proposal: `--data-dir <path>`
  flag (default `./data`) scopes all `load` calls. Any
  absolute or traversing path errors out. Step 001 owns this.
- **BPE performance.** A naive BPE trainer is O(corpus *
  vocab_size * merges). For a ~5 MB corpus at vocab 2048 this
  is still seconds, not minutes, on a laptop. If a demo corpus
  takes longer than ~10s to train we add progress reporting
  and/or early-exit criteria; otherwise leave it simple.
- **`Value::Tokenizer` vs a generic `Value::Handle<T>`.** The
  latter would let Saga 13 reuse the same pattern for trained
  models more easily. Decide in step 004: either `Tokenizer`
  is a dedicated variant, or we introduce a general-purpose
  opaque-value variant now.
- **Experiment writes and the sandbox.** Writing to
  `.mlpl-exp/` from interpreter code is a new side-effect.
  Scope to a `--exp-dir <path>` terminal-REPL flag (defaults to
  `./.mlpl-exp`). Web REPL keeps records in-memory only.
- **Backward compat on `for`.** There is no existing `for`
  construct in MLPL (only `repeat` and `train`). Adding `for`
  is a parser change -- small, but genuinely new syntax. Keep
  it narrow: only `for ident in expr { body }`, no C-style
  variants.
