# Saga: compiler-comparisons-and-read-unwrap

Two upstream compiler rungs the ../demo-file-processing GNU-clone work
(compiled wc / grep / du / ls / cat / head / tail) is blocked on:

1. **Comparisons & logic do not lower.** The compile-to-Rust path
   (`mlpl-lower-rs`) lowers arithmetic binops (`+ - * /` via
   `ApplyBinopExt`) but NOT the comparison/logic builtins, so a
   compiled program using `eq(a, b)` fails with "unsupported construct:
   fncall eq/2". grep needs matching; wc / head / tail need counting
   conditions.

2. **`read_bytes(...)?` does not flow into array ops.** When a
   `read_bytes(p)?` result (an unwrapped `CVal` whose payload is a
   `CVal::Arr(DenseArray)`) is passed onward to an array op that expects
   a `DenseArray`, the generated Rust hits a CVal-vs-DenseArray type
   mismatch. Every file-reading tool hits this.

Interpreter parity is the contract: the compiled comparison ops must
match the interpreter's elementwise 0.0/1.0 semantics and broadcasting,
and the unwrapped-read value must be usable exactly like the
interpreter's `Value::Array`.

This saga does NOT cover streaming / bounded-memory I/O (chunked stdin,
loop-to-EOF) or `for`-loop control flow -- those are the separate
follow-on streaming saga that makes larger-than-RAM processing real.

Each step is TDD (RED failing test -> GREEN lowering -> refactor) with a
gated `MLPL_BUILD_TESTS=1` compiled-binary e2e where an end-to-end
behavior is asserted, plus lower-rs dispatch-coverage updates. Hold or
lower sw-checklist each step.

## Steps

1. lower-comparisons -- lower the elementwise comparison + logic family
   (`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `and`, `or`, `not`, `equal` --
   the exact set the interpreter supports) by mirroring the arithmetic
   binop lowering (a per-op closure through the runtime's elementwise
   trait, or a runtime comparison primitive re-exported from `mlpl-rt`
   analogous to the bit-op family). Match interpreter 0.0/1.0 output +
   broadcasting. Add dispatch-coverage variants. TDD + gated e2e: a
   compiled program computes `eq`, `lt`, `and`, `not` etc. and prints
   the 0/1 results.

2. read-unwrap-array -- fix the `read_bytes(...)?` (and any `?`-unwrapped
   Result whose payload is an array) so the unwrapped value flows into
   array ops without a CVal-vs-DenseArray mismatch. TDD + gated e2e: a
   compiled program `read_bytes(p)?` then reduces / compares the bytes
   (e.g. counts bytes equal to a target -- a wc-newline-count shape) and
   prints the count.

3. docs-close -- document the lowered comparison/logic builtins + the
   read-unwrap fix as compile-to-Rust capable (lang-reference / compiler
   capability doc, WHAT/HOW only). Update docs/future-sagas-queue.md
   (this rung SHIPPED; the streaming saga -- chunked stdin + loop-to-EOF
   bounded reads + `for` -- is the remaining gate for larger-than-RAM
   GNU clones). Refresh docs/companion-demo-file-processing.md. Update
   the wiki errata if a compiled-capability claim flips. Hold
   sw-checklist. `--done`.
