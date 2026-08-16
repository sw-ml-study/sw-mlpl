# Dynamically loaded extensions (B3 design)

Today a native extension is registered STATICALLY: a provider is
compiled into the process and calls `register_c_extension` in-process.
This saga adds DYNAMIC loading -- `dlopen` a provider shared library at
run time and register it through the same C descriptor ABI -- so:

- an extension ships as its own `.dylib`/`.so`, separate from the
  program that uses it;
- a small compiled MLPL app (app + runtime + a compiled MLPL library)
  loads a large extension (e.g. winit/wgpu native3d) at run time
  instead of linking it into one giant binary;
- the same mechanism is general -- native3d is just the first consumer;
  any future extension loads the same way.

Static "one giant binary" linkage is a possible later follow-on, not
this saga's goal.

## Gate (proven)

A throwaway spike confirmed the whole round-trip on macOS: a cdylib
exporting `sw_mlpl_extension_v1` (a `#[no_mangle] extern "C" fn() ->
*const ExtensionDescriptorV1`), `dlopen`ed via `libloading`, its symbol
resolved and called, the descriptor `register_c_extension`ed, and the
function invoked through the registry (`spike:answer()` -> `42.0`). The
existing C descriptor ABI already crosses the `dlopen` boundary
unchanged; the missing piece is only the loader + discovery + trigger.

## Architecture

```
  bin/app  (small: app + mlpl-rt + compiled MLPL library)
    │  load_extension("native3d")  -- an MLPL builtin
    ▼
  loader (mlpl-rt / mlpl-extension-loader)
    │  resolve name -> path via MLPL_EXTENSION_PATH
    │  dlopen(path) [libloading]  ->  resolve sw_mlpl_extension_v1
    │  validate abi_version + struct_size, register_c_extension(desc)
    │  KEEP the library handle alive for the process (no dlclose in v1)
    ▼
  native3d:*  and the Port / UI-host contract resolve as usual
```

- **cdylib packaging.** A provider crate is `crate-type = ["cdylib"]`
  and exports `sw_mlpl_extension_v1` returning a leaked
  `ExtensionDescriptorV1` (lives for the process; the host copies it at
  register time). Its function table is the same the static path uses.
- **The loader.** `libloading::Library::new(path)`; resolve the entry
  symbol; call it; validate the descriptor (`abi_version`,
  `struct_size`); `register_c_extension`; `mem::forget`/store the
  `Library` so the code stays mapped (fail-closed on any error).
- **Discovery.** `MLPL_EXTENSION_PATH` (colon-separated directories) is
  searched for the platform filename (`libnative3d.dylib` on macOS,
  `libnative3d.so` on Linux, `native3d.dll` on Windows); a logical name
  maps to that filename. A default directory (next to the binary or an
  install dir) is the fallback. This is the path a `justfile`/run
  script sets.
- **The trigger.** A builtin `load_extension(name_or_path)` (works in
  the interpreter and, later, in compiled code). The `use native3d`
  facade (B1) can layer on later; an explicit builtin is the first cut.
- **Interpreter first.** The loader + builtin land in the interpreter
  path and are proven headlessly (mlpl-repl loads a test cdylib by name
  and calls it). Compiled binaries reuse the same loader; the compiler
  work to lower `load_extension` + extension/Port calls is the
  immediate FOLLOW-ON saga, not this one.

## Division of labor

- **sw-mlpl** provides the loader, `MLPL_EXTENSION_PATH` discovery, the
  `load_extension` builtin, cdylib validation/versioning, and the
  packaging convention + docs.
- **The provider** (demo-extensions native3d) builds itself as a cdylib
  exporting `sw_mlpl_extension_v1`; its window/GPU/event-loop code is
  unchanged (it already implements the shipped Port/UI-host contract).

## Non-goals (separate or later)

- `dlclose` / true unload / hot-reload (hold the handle for the
  process; v1 never unloads).
- Manifest + search-path trust/verification resolver (A7).
- The `use`/facade import grammar (B1).
- Compiling `load_extension` + extension/Port calls into native code
  (the compiler-parity FOLLOW-ON that makes the small compiled app real).
- Static "one giant binary" linkage (possible later follow-on).

## Steps (TDD each)

1. **loader-gate** -- land a kept regression of the spike: an in-repo
   test cdylib exporting `sw_mlpl_extension_v1` + a test that dlopens,
   registers, and invokes it. Proves the mechanism in CI.
2. **dynamic-loader** -- the loader function (dlopen + resolve +
   validate + register + hold handle) with clear errors, reused by the
   interpreter and (later) compiled code.
3. **discovery-and-builtin** -- `MLPL_EXTENSION_PATH` name->path
   resolution + the `load_extension(name_or_path)` builtin wired into
   the interpreter. TDD: set the env var, load a test cdylib by name,
   call its function from MLPL.
4. **cdylib-packaging-and-contract** -- the provider cdylib convention
   (crate-type, entry export, filename mapping) + docs; a headless
   provider-shaped test cdylib (arrays/records/handles, not just a
   scalar) proving the boundary through `dlopen`.
5. **docs-close** -- user + contract docs, wiki errata, demo-extensions
   upstream note; queue the compiler-parity follow-on (lower
   `load_extension` + extension/Port calls) and the dlclose / manifest /
   use-facade follow-ons. `--done`.
