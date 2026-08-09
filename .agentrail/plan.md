# Saga: extensions-c-abi-adapter

demo-extensions request: their provider exports a #[repr(C)] C
descriptor (sw_mlpl_extension_v1 -> *const ExtensionDescriptorV1),
but the shipped static registry takes a SAFE-Rust
ExtensionDescriptorV1. Add a host adapter that accepts their C
descriptor and registers it -- distinct from dynamic loading
(this is descriptor SHAPE, not dlopen). Match their exact C
layout (demo-extensions/crates/mlpl-extension-abi/src/model.rs)
byte-for-byte so a real (statically linked) provider registers
unchanged; sw-mlpl publishes the canonical layout (it owns
registration).

Enabling change: the registry's ExtFn is a bare `fn` pointer,
which cannot capture a per-function C trampoline. Change it to a
boxed closure (Arc<dyn Fn(&[ExtValue]) -> Result<ExtValue,
ExtError> + Send + Sync>) so both the existing safe static
providers AND the C adapter (capturing the C invoke ptr) register
through one path.

Scope: SCALAR V1 values only (nil/bool/i64/f64/utf8/bytes -- the
ExtValue set); DenseArray + NativeHandle AbiValue variants are the
deferred arrays-handles saga (err on them for now). Static
linking only (no dlopen).

## Steps
1. boxed-extfn -- ExtFn -> Arc<dyn Fn..>; update call_contained,
   ExtFnDesc (drop Debug derive / manual), registry, hello
   provider, fncall_ext/ext_marshal; keep all extension tests
   green (regression).
2. cabi-crate -- new mlpl-extension-cabi: #[repr(C)] V1 structs
   (AbiSlice/ValueTag/AbiValue/ValuePayload/FunctionDescriptorV1/
   ExtensionDescriptorV1/AbiErrorV1/InvokeFnV1) matching
   demo-extensions; register_c_extension(*const descriptor)
   validates (struct_size/abi_version/bounds/utf8/dup) + wraps
   each C invoke in a closure marshaling ExtValue<->AbiValue with
   catch_unwind + AbiErrorV1 -> registers. TDD: an in-test C
   provider (extern "C" invoke) registers via register_c_extension
   and hello:answer() -> 42 through the interpreter.
3. close -- expose register_c_extension for downstream; docs
   (companion-demo-extensions: C adapter shipped), wiki, q-and-a
   (demo-extensions can register its C provider), --done.
