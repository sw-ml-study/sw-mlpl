# Saga: extensions-dynamic-load

Add DYNAMIC loading of native extensions: `dlopen` a provider shared
library at run time and register it through the existing C descriptor
ABI. So an extension ships as its own `.dylib`/`.so`, and a small
compiled MLPL app loads a large extension (winit/wgpu native3d) at run
time instead of linking one giant binary. The mechanism is general --
native3d is the first consumer. Static "one giant binary" linkage is a
possible later follow-on, not this saga.

Full design: `docs/extensions-dynamic-load-design.md`.

## Gate (PROVEN)

A spike confirmed the whole round-trip on macOS: a cdylib exporting
`sw_mlpl_extension_v1` (`#[no_mangle] extern "C" fn() -> *const
ExtensionDescriptorV1`), `dlopen`ed via `libloading`, resolved, called,
`register_c_extension`ed, and invoked through the registry
(`spike:answer()` -> `42.0`). The C descriptor ABI already crosses the
`dlopen` boundary unchanged; the missing piece is the loader +
discovery + trigger.

## What exists

- `mlpl-extension-cabi`: the `#[repr(C)]` V1 descriptor +
  `ExtensionEntryV1 = fn() -> *const ExtensionDescriptorV1` (the cdylib
  export type) + `register_c_extension` + descriptor validation.
- `mlpl-extension-registry`: process-global registration + lookup.
- `libloading` is already resolved in the eval/cli/native-rt
  workspaces (cached).
- Only STATIC in-process registration is wired today.

## Non-goals (separate or later)

- `dlclose` / true unload / hot-reload (hold the handle for the
  process; v1 never unloads).
- Manifest + trust/verification resolver (A7).
- The `use`/facade import grammar (B1).
- Compiling `load_extension` + extension/Port calls to native code (the
  compiler-parity FOLLOW-ON that makes the small compiled app real).
- Static one-giant-binary linkage.

## Steps (TDD each)

1. **loader-gate** -- land a kept regression of the spike: an in-repo
   test cdylib exporting `sw_mlpl_extension_v1` + a test that dlopens,
   registers, and invokes it (proves the mechanism in CI, cross-
   platform filename aware).
2. **dynamic-loader** -- the loader function (dlopen + resolve entry +
   validate abi_version/struct_size + register + hold the `Library`
   handle) with clear errors, reused by the interpreter and later the
   compiler. TDD: load the test cdylib by path.
3. **discovery-and-builtin** -- `MLPL_EXTENSION_PATH` name->path
   resolution (logical name -> `lib<name>.dylib`/`.so`/`native<name>.dll`
   + default dir) and a `load_extension(name_or_path)` MLPL builtin
   wired into the interpreter. TDD: set the env var, load by name from
   MLPL, call the extension function.
4. **cdylib-packaging-and-contract** -- the provider cdylib convention
   (crate-type, entry export, filename mapping) + docs; a headless
   provider-shaped test cdylib exercising arrays/records/handles across
   `dlopen`, not just a scalar.
5. **docs-close** -- user + contract docs, wiki errata, demo-extensions
   upstream note; queue the compiler-parity follow-on and the dlclose /
   manifest / use-facade follow-ons. `--done`.
