# Memory Model

How does sw-MLPL manage memory? There is no manual
`alloc`/`free`, no garbage collector, and no borrow checker
that the MLPL programmer ever sees. This document explains
why none of those are needed.

## TL;DR

sw-MLPL is a tree-walking interpreter written in Rust. It has
no memory model of its own -- every MLPL value is a plain,
owned Rust value, so MLPL inherits Rust's RAII / `Drop`-based
management wholesale. Memory is freed deterministically when
the owning Rust value goes out of scope. No tracing GC and no
reference counting are required because the value graph is an
acyclic tree of owned data, and the borrow checker is
invisible to MLPL users because values have copy semantics,
not aliasing.

## Values are owned Rust data

The runtime value type lives in
`components/eval-types/crates/mlpl-eval-types/src/value.rs`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Array(DenseArray),          // dense numeric tensor
    Str(String),
    Model(ModelSpec),
    Tokenizer(TokenizerSpec),
    BuiltinRef { name: String },
    DeviceTensor { peer, handle, shape, device },
    Record { fields: BTreeMap<String, Value> },
    StrList { items: Vec<String> },
    Result { ok: bool, payload: Box<Value> },
}
```

`DenseArray`
(`components/array/crates/mlpl-array/src/dense.rs`) is just:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct DenseArray {
    shape: Shape,
    data: Vec<f64>,
    labels: Option<Vec<Option<String>>>,
}
```

There is no `Rc`, no `Arc`, no `RefCell`, and no `unsafe` in
the value path. The `Vec<f64>` backing each tensor IS the heap
allocation. A `Value` is therefore a self-contained owned tree:
a `Record` owns its fields by value, a `Result` owns a
`Box<Value>`, a `StrList` owns its `Vec<String>`, and so on.

Variables live in the `Environment`
(`components/eval/crates/mlpl-eval/src/env.rs`), which is a
bundle of `HashMap`s keyed by name -- `vars: HashMap<String,
DenseArray>`, `strings`, `records`, `string_lists`, `results`,
`user_fns`, and friends. Each map owns the values it holds.

## Why no manual alloc / free

Rust's RAII frees memory deterministically when the owner is
dropped. In the interpreter that happens at well-defined
points:

- **Reassignment.** `x = expr` calls `HashMap::insert`, which
  returns the previous binding; that old `DenseArray` is
  dropped immediately and its `Vec<f64>` is freed.
- **End of a user-defined function call.** `call_user_fn`
  (`components/eval/crates/mlpl-eval/src/eval_user_fn.rs`)
  saves the prior bindings for each parameter name, runs the
  body, then restores the saved values or `remove_var`s the
  parameter. The call's parameter temporaries drop at that
  point.
- **Intermediate expression results.** Values produced while
  evaluating an expression are owned on the Rust call stack
  and drop at the end of the statement that produced them.
- **Session teardown.** When the `Environment` is dropped (REPL
  exit, end of a script run), every map drops and frees its
  contents transitively.

The MLPL programmer never writes `free`. There is nothing to
free manually because every allocation has exactly one owner
whose scope determines its lifetime.

## Why no garbage collector

A tracing GC (or reference counting) exists to reclaim two
things that sw-MLPL cannot create:

1. **Cycles.** A cycle requires shared, mutable, indirected
   references (`Rc<RefCell<...>>` or pointers). MLPL values
   are owned by value with no indirection a user can alias, so
   a value cannot refer back to itself or to an ancestor. The
   ownership graph is a tree by construction -- it is
   impossible to build an infinite or cyclic owned structure.
2. **Shared heap graphs with unclear ownership.** Every value
   has exactly one owner (the binding, the enclosing
   container, or the stack frame). When that owner drops, the
   value drops.

Because the graph is acyclic and single-owner, `Drop` alone is
a complete and precise reclamation strategy. There is no
collector thread, no pause, and no allocation-rate-driven
overhead.

## Why no borrow checker (for MLPL users)

MLPL uses **value semantics**: `Value` derives `Clone`, and
binding `y = x` clones the underlying data (the whole
`Vec<f64>` of a tensor). Two MLPL variables never alias the
same buffer. Function parameters are bound to evaluated
argument values, shadowing any outer variable of the same name
for the duration of the call and restoring it afterward (see
`call_user_fn`), so there is no shared mutable state for a user
to reason about.

With no aliasing, there are no borrow-checking obligations at
the language level: there is no way to hold two references to
the same mutable buffer, so use-after-free, data races, and
iterator invalidation are not expressible in MLPL.

The borrow checker is, of course, fully active inside the
interpreter's own Rust source -- it is what guarantees the
host implementation is sound. It is simply invisible to the
person writing MLPL, who only ever sees independent owned
values.

## The trade-off: copy-on-assign

Value semantics cost copies. Binding a large tensor clones its
entire `Vec<f64>` -- a `[1000, 1000]` matrix copies 8 MB on
assignment. The interpreter reduces this where it can:

- Internal operations borrow `&DenseArray` rather than cloning
  (`Value::as_array`).
- The last use of a value can move it out instead of cloning
  (`Value::into_array`).

But the user-facing contract stays simple: each binding owns
its own data. For an array language used for teaching and
experimentation this is a deliberate trade -- predictable,
explainable semantics over zero-copy performance. A future
copy-on-write layer (e.g. `Arc<Vec<f64>>` with clone-on-mutate)
could cut the copies without changing observable behavior; it
is intentionally not in v0 because it would add the very
shared-ownership machinery this model avoids.

## Device tensors: the one off-host case

`Value::DeviceTensor` is the exception that proves the rule. It
holds only metadata -- `{ peer, handle, shape, device }` -- and
the actual bytes live off-host on an accelerator peer (e.g.
MLX). That memory is not managed by the host's `Drop`; it is
owned by the peer and reclaimed through the handle's lifecycle.
Materializing it back to host memory via `to_device("cpu", x)`
produces an ordinary owned `Value::Array` that rejoins the RAII
model described above.

## See also

- `docs/architecture.md` -- crate / component layout
- `components/eval-types/crates/mlpl-eval-types/src/value.rs`
  -- the `Value` enum
- `components/array/crates/mlpl-array/src/dense.rs` --
  `DenseArray`
- `components/eval/crates/mlpl-eval/src/eval_user_fn.rs` --
  parameter save / restore on UDF calls
