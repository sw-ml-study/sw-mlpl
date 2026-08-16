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
- [~] **compiler-byte-io** -- IN PROGRESS. Shipped: the bit ops
      (`band`/`shr`/`bmask`/...), a loud-reject `array_to_bytes`
      validator + `write_stdout` parity (rejects, returns
      `ok(count)`/`err`, propagates I/O errors), and `read_bytes`
      (whole + range, EOF-clamped) + `file_size` returning Results,
      sandboxed to `MLPL_FS_ROOT`/cwd via the interpreter's
      `contained` check. Remaining rungs: `write_bytes`/`append_bytes`
      and the text conversions (`tokenize_bytes`/`decode_bytes`/
      `to_int`).
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
- [x] **extensions-arrays-handles (B4/B5)** -- dense-array marshaling
      at the C boundary (call-lifetime rooting, both directions) +
      opaque native-handle values (mint-on-return, provider-validated)
      + structured record returns (named fields, nested). The data
      boundary for the interpreted interactive native-3D demo; only
      the event loop (B6) remains for the live loop.
- [ ] **extensions-dynamic-load (B3)** -- `dlopen` a provider,
      resolve `sw_mlpl_extension_v1`, validate + register; residency
      + atomic-load + quiescence-for-unload policy.
- [ ] **extensions-package-trust (B7)** -- manifest + search-path +
      trust resolver (needs B3).
- [x] **extensions-event-loop (B6)** -- SHIPPED. The Port primitive
      (share-nothing command/event channels), `on`/`off`/`run` handler
      dispatch, bounded `port_poll(port, limit)`, the parked-main launch
      inversion (interpreter on a worker, UI host on the main thread),
      and the provider contract (the provider plugs in as the Rust
      UI-host closure). The INTERPRETED interactive native-3D app works
      end to end on these primitives. See docs/ports-and-applets.md and
      docs/extensions-event-loop-design.md.
- [ ] **extensions-compiler-parity (B2)** -- compile an extension +
      event-loop applet to a self-contained native binary. Needs
      compiler/runtime parity with the interpreted host path: compiled
      provider registration, parked-main launch inversion in generated
      code, compiled Port + handler parity, value-boundary parity
      (arrays / records / errors / handles across compiled provider
      calls), module/include packaging, native provider+winit/wgpu
      linkage (no unstable ABI), target-aware macOS/Linux packaging, and
      deterministic teardown. Scoped from
      demo-extensions/docs/compile-3d-app-blocked.md; queued in
      docs/future-sagas-queue.md.

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
