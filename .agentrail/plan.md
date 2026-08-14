# Saga: extension-arrays-handles

Open the sw-MLPL native-extension boundary beyond scalars, so the
interactive native-3D demo (demo-extensions) can run its MLPL loop
against a real `native3d:*` provider. The renderer is not blocked; the
missing piece is the live extension boundary.

Downstream blockers addressed: B4 (dense arrays at the boundary, both
directions) + B5 (native handles) + structured record/array returns.
The follow-on event-loop (B6) and compiler-parity (B2) are SEPARATE
sagas; B2 is not needed for the first interpreted demo.

## What exists (survey)

- Safe boundary `mlpl_extension_abi::ExtValue` = scalar-only
  {Nil,Bool,I64,F64,Str,Bytes}; `ExtFn = Fn(&[ExtValue]) -> Result<
  ExtValue, ExtError>`; `call_contained` catches panics.
- C wire ABI `mlpl_extension_cabi::model`: `ValueTag` has DenseArray=6,
  NativeHandle=7 (DECLARED, rejected); `AbiArrayView{dtype,rank,data,
  shape,strides}`, `DTypeTag{U8,I64,F32,F64}`, `AbiHandle{extension_id,
  type_id,slot,generation}` all declared but not marshaled.
- `register_c_extension` + `mlpl_extension_registry::register` = one
  path for static + C providers. Dispatch: eval.rs -> fncall_ext.rs ->
  ext_marshal.rs (to_ext/from_ext/finish), scalar-only today.
- Proven tests: extension_dispatch_tests.rs, extension_cabi_dispatch_
  tests.rs (a #[repr(C)] `cabi_demo` provider, arity-0 scalar).

## Design

- Extend the SAFE `ExtValue` with `Array { dtype, shape, data }` and
  `Handle(AbiHandle-equivalent)` variants (Rust enum; C ABI already has
  the tags). Dtypes: u8/i64/f32/f64 (per dense-array-views.md); rank
  1-8; contiguous row-major; <=16 MiB.
- Marshal both layers: `mlpl-eval/ext_marshal` (Value <-> ExtValue) and
  `mlpl-extension-cabi/marshal` (ExtValue <-> AbiValue/AbiArrayView/
  AbiHandle). Reject unsupported dtype/rank/stride/size with clear
  errors.
- Native handles: a `Value` kind carrying an opaque handle
  (extension_id,type_id,slot,generation); the registry validates
  extension/type/slot/generation/activity on use; ordinary numeric
  construction cannot forge one.

## Steps (TDD each)

1. **array-args-in** -- `ExtValue::Array`; `ext_marshal::to_ext`
   accepts a `Value::Array` (rank>=1, contiguous) -> ExtValue::Array;
   `cabi/marshal` writes an `AbiArrayView` (tag 6). Test: a repr(C)
   provider `sum_array` receives an f32 `[2,3]` of 1..6 and returns
   21.0 in ONE call. Reject: unsupported dtype/rank/oversize.
2. **array-results-out** -- `from_ext`/`cabi` read an `AbiArrayView`
   returned BY a provider -> `Value::Array`. Test: a provider returns
   an `[N,3]` array; MLPL gets the array back.
3. **native-handles** -- `ExtValue::Handle` + a `Value` handle kind;
   round-trip through a registry-side slot table with generation +
   type validation. Test: a provider `open` returns a handle, a later
   `use_handle(h)` accepts it; stale/forged/wrong-type/cross-extension
   handles error.
4. **structured-returns** -- provider returns a RECORD (or list of
   records) for event batches (`poll_events`). Test: a provider returns
   `{kind, x, y}` records; MLPL reads the fields.
5. **docs-catalog-close** -- lang-reference + extension contract docs;
   update demo-extensions upstream-contract note; wiki errata; queue
   the event-loop (B6) + compiler-parity (B2) follow-on sagas. Rebuild
   release binary. --done.

## Non-goals (separate sagas)

- Event-loop / main-thread ownership / poll_events window semantics (B6).
- Dynamic loading (dlopen / sw_mlpl_extension_v1) (B3), package trust (B7).
- `use`/facade (B1), compiler provider-registration parity (B2).
