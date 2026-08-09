# Companion: demo-extensions (native extension SDK)

PLANNING doc (forward-looking). Records the upstream contract
demo-extensions asks of sw-mlpl, extracted from
`../demo-extensions/docs/` (binding list:
`upstream-contract.md`; ABI: `abi-v1.md`; metadata:
`signature-metadata.md`; packages: `extension-packages.md`;
acceptance: `foundation-acceptance.md`). Companion index:
`companion-repos.md`.

## What demo-extensions is

A downstream repo building a **native/dynamic Rust extension
ecosystem** for sw-MLPL -- an ABI + SDK + loader + package
format, proven first by a headless `hello` extension, later a
native 3D (wgpu/winit) stack. It may not modify sw-mlpl. Its
programming model is "an MLPL module, not an `ffi.call` API": a
Rust `cdylib` exports one C-ABI symbol; a loader validates it and
registers it under a PRIVATE native namespace (`_hello`); a
shipped `module.mlpl` facade re-exposes it under a PUBLIC
namespace (`hello`); the user writes `use hello` and calls
ordinary-looking MLPL functions.

Layering (demo-extensions owns everything below the host line):

```
MLPL module facade (module.mlpl)     <- public `hello.*`
  -> [HOST] extension registry + import
  -> [HOST] compiler / static-provider hook
  -> package resolver + dynamic loader (downstream)
  -> versioned C ABI V1 (downstream)
  -> safe Rust SDK/macros (downstream)
  -> extension impl (downstream)
```

The reported blocker: demo-extensions has proven ITS side (a
Rust harness loads a real independently-built `.dylib`, validates
ABI V1, invokes namespaced functions, contains panics), but
sw-mlpl exposes NO public host API to accept that validated
descriptor and make its namespace reachable from the REPL,
interpreted scripts, and compiled programs. A local mock is
explicitly "never evidence".

## The upstream asks (host side only)

### Public registry

- **A1 -- public registration API** (LARGE): a host API that
  accepts a validated `ExtensionDescriptorV1` and registers its
  functions + native value types into an importable namespace,
  via ONE common path for both a dynamically loaded library and a
  statically linked provider. Fail-closed; copy metadata into
  host-owned memory; retain no extension-owned pointers.
- **A2 -- versioned value/error boundary** (MEDIUM): a public
  boundary value type not exposing evaluator internals. V1
  scalars: nil / bool / i64 / f64 / UTF-8 string / bytes.
  Extension errors -> a host error; panics contained (never
  unwind through the host). `#[repr(C)]`, fixed-width tags,
  reserved-must-be-zero, ABI-version + struct-size negotiation.
- **A3 -- native-handle value** (MEDIUM, DEFERRED): an MLPL value
  carrying `NativeHandle { extension_id, type_id, object_id }`
  (generational id) with deterministic finalization; no raw
  pointer is ever an MLPL value. Later saga (with arrays), not
  the first slice.

### Import

- **A4 -- `use <package>` import** (LARGE): on `use hello`,
  resolve the package, load the library, check ABI, register
  native functions/types, then load `module.mlpl` AFTER native
  registration and publish the public namespace. Manifest
  (`extension.toml`) declares public name, semver, ABI version,
  facade, private namespace, one artifact per target triple;
  resolution rejects unsupported ABI/triple, duplicates,
  non-normal path components, symlink escapes, missing artifacts,
  namespace drift; never searches the CWD implicitly.
- **A5 -- help/signature metadata** (SMALL-MEDIUM): extension
  signatures/docs surfaced to MLPL `help` / `:describe` /
  `:builtins`. Metadata is one bounded UTF-8 TOML doc:
  `[[functions]]` (`name`/`documentation`/`returns`/ordered
  `[[functions.arguments]]`) + `[[types]]`. Plus actionable
  argument/type/shape diagnostics.

### Compiler hooks

- **A6 -- registration parity across REPL / interpreted /
  compiled** (LARGE): equivalent hooks for all three modes;
  compiled programs use a STATIC provider first (the compile path
  ships no interpreter, so no runtime loading). `use native3d`
  must behave identically across REPL, script, compiled+dylib,
  and fully-linked exe, without changing MLPL source.
- **A7 -- search paths + trust/integrity policy** (MEDIUM):
  deterministic search paths, platform triples, manifest
  validation, trust policy; optional `:extension load/list/info/
  unload`. Security half deferred to a later saga.

### Interaction (out of scope until native3d)

- **A8 -- host event-loop service** (LARGE): winit main-thread /
  repeated-REPL-eval policy, bounded event delivery, reentrancy
  rules. Needed only at the live-viewer saga.

## Sequencing (per the contract)

A2 + A1 first (everything registers through the registry) ->
static-provider parity is the chosen FIRST concrete step ->
import/`use` (A4) + facade load depend on the registry -> help
(A5) depends on the registry's host-owned model -> compiler
static hook (A6) reuses the same registration path. A3 (handles)
and A8 (event loop) are explicitly later.

## Do not build twice (overlaps with existing sw-mlpl direction)

- **Builtin dispatch is compile-time, not a runtime registry.**
  `mlpl-runtime/src/builtins.rs` is a hard-coded `try_call`
  chain + name `match`; there is no runtime registration today.
  A1 is genuinely new, but the new registry should SUBSUME/wrap
  this static chain so builtins and extensions share one lookup
  -- not a second parallel dispatch.
- **`mlpl-builtin-catalog` is static help metadata.** A5 should
  feed the SAME `help` / `:describe` / `:builtins` surface, not a
  parallel one.
- **`include` is source-text splicing**, a different mechanism
  from `use` (native load + register + publish). Keep `include`
  separate but reconcile the `use` keyword design with it.
- **The queued `modules-namespaces` saga directly overlaps
  A4/A5** (qualified names, exports, private helpers -- exactly
  the `hello` / `_hello` public/private split). Build the
  import/namespace foundation ONCE to serve both.
- **Compile-to-Rust constrains A6 (compatibly):** MLPL has no
  `eval`/`exec`, so the compiled path needs a STATIC provider
  (which is why the contract asks for exactly that first); the
  `mlpl-lower-rs` backend is the insertion point. No conflict,
  but the compiled hook must be link-time, never a runtime
  parser.

## Recommended minimal first upstream saga

**`extension-registry-static-provider`** -- a public host
registration + import path proven first via a STATICALLY LINKED
provider (sidesteps dynamic loading, arrays, handles, and the
event loop). Smallest slice that flips a blocked acceptance row
to "proven by a named test":

1. A2 minimal: public boundary value/error for the V1 scalars
   (nil/bool/i64/f64/string/bytes) + contained panics. No arrays,
   no handles.
2. A1 minimal: public API taking an already-validated descriptor
   (validation stays downstream), registering namespaced
   zero-/scalar-arg functions; wired as a lookup the existing
   `builtins.rs` chain defers to (no double dispatch).
3. A6 static provider FIRST: register via a statically linked
   provider -- no dynamic load or event loop.
4. A4 minimal `use`: `use hello` resolves the registered
   namespace and loads the `module.mlpl` facade after
   registration so public `hello.*` calls `_hello.*`. Reconcile
   the keyword with `modules-namespaces` / `include` first.
5. A5 minimal: surface the registered TOML signature through
   `help` / `:describe`, reusing `mlpl-builtin-catalog`.

**Acceptance** (mirrors demo-extensions' own gate): from an
interpreted `.mlpl` script and the REPL, `use hello` then a call
returns the typed `i64` `42`, a failing call returns a typed MLPL
error, a panicking extension is contained, and `help` shows the
declared signature -- with the provider statically linked.
Dynamic loading, arrays, native handles, the event loop, and the
compiled `--embed-extension` path are out of this first slice.

## Status

Contract recorded; the first-slice saga is queued in
`future-sagas-queue.md`. Not yet started -- it is a substantial
architectural item (a runtime registry that rationalizes the
existing static dispatch) and should be sequenced deliberately
with `modules-namespaces`, not bolted on.
