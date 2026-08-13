# Saga: typed-packed-bytes

A typed, packed byte-buffer value for sw-MLPL: real element dtypes,
observable storage footprint, bit-level reinterpretation, and typed
little-/big-endian readers. Unblocks two downstream repos:

- **demo-memory** (HARD blocker "packed layout with observable size"):
  needs `size_bytes(x)` on packed storage to make bytes-per-key /
  locality claims measurable (Bloom/counting filters, tiny-pointer
  navigation, LRU/KV eviction).
- **demo-ml-utils** (#2, P0 "typed byte arrays / reinterpret"): needs
  real dtypes (u8..f64), `reinterpret(bytes, dtype)` without numeric
  conversion, and typed readers (`read_u32_le`, ...) for Safetensors /
  GGUF / quantization headers.

## Design

Introduce a NEW value kind `Value::Bytes { dtype, data: Vec<u8> }` (a
packed byte buffer tagged with an element dtype) instead of retrofitting
the f64-backed `DenseArray`. This keeps the tensor / autograd / viz path
(all f64) untouched and adds a separate systems-data path. Little-endian
is the canonical packing; big-endian is an explicit reader family.
Bounded, index/offset-only (no pointer arithmetic), Result-valued on
out-of-range access.

Follow code_metrics.md: new packed-bytes logic lives in named files
(model / pack / read / etc.), facade lib.rs, pure free functions, define
the typed reader family once via a macro/table (loose-coupling
"define once, invoke many").

## Steps

1. **bytes-value-and-pack** -- add `Value::Bytes { dtype, data }` with a
   `ByteDtype` enum (u8,i8,u16,i16,u32,i32,u64,i64,f32,f64), value_kind +
   Display (`<bytes: u8[N]>`), wire the exhaustive Value matches, and a
   `pack(array, "dtype") -> bytes` builtin (canonical little-endian).
   TDD.
2. **size-bytes** -- `size_bytes(x) -> scalar` packed footprint in bytes
   (Bytes = data.len(); f64 Array = elem_count*8; record/strlist = sum).
   demo-memory's hard-blocker probe. TDD.
3. **reinterpret** -- `reinterpret(bytes, "dtype") -> bytes` re-views the
   same bytes under a new dtype (byte length must divide the new width;
   no numeric conversion). demo-ml-utils. TDD.
4. **typed-readers-le** -- `read_{u,i}{8,16,32,64}_le` + `read_f{32,64}_le`
   `(bytes, offset) -> scalar`, bounds-checked (Err on OOB). Define the
   family once via a macro/table. TDD.
5. **typed-readers-be-and-unpack** -- big-endian `_be` reader family +
   `unpack(bytes, "dtype") -> array` (numeric widening to f64, inverse of
   pack). TDD.
6. **docs-catalog-close** -- lang-reference "Typed byte buffers" section,
   glossary + mlpl-builtin-catalog entries, wiki errata, downstream
   acceptance notes, rebuild target/release/mlpl-repl, queue follow-ons
   (f16/bf16, endian writers, streams), --done.

## Non-goals (queued follow-ons)

- f16 / bf16 dtypes (need a half-float codec).
- Mutating typed writers (`write_u32_le` at offset into a buffer).
- The stream/fold abstraction (demo-ml-utils #3) -- separate saga.
- Compiler (`mlpl build`) lowering of the bytes family -- compiler-parity
  track.
