# Downstream unblock plan (saga checklist)

A cross-repo roadmap: the sagas that unblock the four companion
repos (`demo-file-processing`, `demo-extensions`, `demo-algorithms`,
`demo-ml-utils`). Each item is one agentrail saga (or a step
within one). Ordering follows dependencies, not wishlist size.

Legend: [x] shipped -- [~] in progress -- [ ] queued.
Full per-saga notes live in `docs/future-sagas-queue.md`; the
per-repo contracts in `docs/companion-demo-*.md`.

## Track A -- Compiler parity (unblocks demo-file-processing + demo-extensions B2)

The critical path. Every real byte/format app compiles the same
MLPL the interpreter already runs; the compiler surface is the gate.

- [x] **compiler-source-loading** -- `include` resolved via the
      interpreter's `expand()` + FS sandbox.
- [x] **compiler-functions** (param-only) -- `def u:` -> nested Rust
      fn; `u:name(args)` calls; free/global read rejected.
- [x] **compiler-control-flow (if/else)** -- `if` over DenseArray
      truthiness; real Rust `return` (return-in-branch exits the fn).
- [x] **compiler-control-flow (while + mutable vars)** -- first
      `Assign` -> `let mut`, rebind -> reassign; `while` -> Rust
      `while`. (`for` still deferred -- needs row-extraction / `take`.)
- [x] **compiler-records-results** -- `RecordLit` + `FieldAccess` +
      `ok`/`err` + `?`/`check`. Records lower to `CVal::Record`,
      field access unwraps, `ok`/`err` -> `CVal::Result`. A body that
      produces `ok`/`err`/a record (or uses `?`) lowers to `-> CVal`;
      `?` unwraps an `ok` payload or early-returns the `err`. The real
      pattern compiles + runs: `f = u:fit(n)?; f.slope` where `u:fit`
      returns `ok({...})`. (Deferred: `for`, nested-record field
      chains, arithmetic on a `?`-unwrapped value, top-level `?`.)
- [ ] **compiler-byte-io** -- lower `read_bytes` (whole + range),
      `file_size`, `append_bytes`/`write_bytes` + the array/bit/text
      ops the demos use. MUST share validation + error semantics
      with the interpreter (reject invalid bytes, don't coerce;
      propagate sink write/flush errors, don't discard).
- [ ] **compiler-process-semantics** -- drop the auto-printed
      numeric result trailer (pristine binary stdout); lower
      `print`/`eprint`/`exit`/`read_stdin`; useful exit status.
- [ ] **compiler-parity-capstone** -- positive byte + format artifact
      parity (byte-identical output, reparse, malformed-file status,
      bounded RSS) + a source-free audit. Closes demo-file-processing.
- [ ] **extensions-compiler-parity (B2)** -- a link-time
      static-provider registration hook in generated `main` calling
      the same registry, so extension calls behave identically in
      compiled binaries. RIDES the rungs above.

## Track B -- Extensions (unblocks demo-extensions)

- [x] **extensions-c-abi-adapter** -- `register_c_extension` accepts
      a `#[repr(C)]` provider descriptor (scalar values).
- [ ] **extensions-c-help-metadata (B8)** -- SMALL: parse the
      descriptor's TOML metadata (returns/documentation scan) into
      the help catalog + validate name/arity. `:describe _hello:answer`
      renders the signature. Self-contained in `mlpl-extension-cabi`.
- [ ] **extensions-use-facade (B1)** -- `use hello` + dotted
      `hello.answer()` public facade; private `_hello:*` hidden.
      Built with the modules/namespaces language surface.
- [ ] **extensions-arrays-handles (B4/B5)** -- dense-array marshaling
      at the C boundary (call-lifetime rooting) + native handle
      values (ownership + finalization).
- [ ] **extensions-dynamic-load (B3)** -- `dlopen` a provider,
      resolve `sw_mlpl_extension_v1`, validate + register; residency
      + atomic-load + quiescence-for-unload policy.
- [ ] **extensions-package-trust (B7)** -- manifest + search-path +
      trust resolver (needs B3).
- [ ] **extensions-event-loop (B6)** -- host policy for native
      windows / event delivery (needs B5 + main-thread/reentrancy).

## Track C -- Codec (unblocks demo-algorithms)

- [x] **native-codec** + **codec-mlpb-integrity (v2 CRC32)**.
- [ ] **codec-streaming** -- incremental (chunked) encoder/decoder.
      demo-algorithms' CURRENT blocker.
- [ ] **codec-reference-tables** -- shared-ref dedup + cycle policy.
- [ ] **codec-toml-tagged** -- TOML `$mlpl` tagged mode.
- [ ] **codec-migration-hooks** -- version migration + path-aware errors.
- [ ] **codec-numeric-types** -- non-f64 element arrays at the boundary.

## Track D -- Real-model training (unblocks demo-ml-utils / Qwen)

Split by language-primitive vs model-specific integration (the
latter belongs in an extension/library, not the core language):

- [ ] **CORE, if a generic primitive is missing** -- generic weight
      save/load (adapter persistence), generation/`GenState` gaps,
      generic LoRA-layer gaps. (Tokenizer ops, autograd, Model DSL,
      `lora`/`clone_model` already exist -- audit first.)
- [ ] **EXTENSION/LIBRARY (not core)** -- quantized causal-model
      loader + a specific pretrained (Qwen) architecture + quantized
      formats. Rides the extensions track (C-ABI provider or a
      companion Rust crate). Do NOT bake a specific model loader
      into the language.

## Recommended sequence

1. **Track A** to completion (control-flow -> records/Results ->
   byte-IO -> process-semantics -> capstone). Unblocks
   demo-file-processing AND enables extensions B2.
2. **extensions-c-help-metadata (B8)** whenever a small self-contained
   win is wanted (independent of everything).
3. **codec-streaming** for demo-algorithms (independent track).
4. **Track B** language/loader sagas (facade, dynamic load,
   arrays/handles) as their own focused efforts.
5. **Track D** last (largest; audit core coverage first, then push
   the model-specific loader into an extension/library).

## Runtime handles (separate demo-file-processing gate)

- [ ] **runtime-sink-handle** / **runtime-source-handle** -- binary
      stdin + persistent source/sink handles (backpressure,
      flush/close lifecycle, memory high-water). Distinct from the
      compiler track; needed for true stdin-driven streaming.
