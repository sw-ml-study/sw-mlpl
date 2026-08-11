# Saga: compiler-control-flow

demo-file-processing's real byte/format apps now pass include +
user-function lowering and stop at unsupported `If`. Lower control
flow, in slices (the hardest rung; interpreter-equivalent semantics
are the whole point).

Key semantics: the interpreter's `if` is truthy iff a non-zero
scalar or Ok(_). Branches that early-`return` must exit the
ENCLOSING function -- so the compiler-functions "trailing return =
tail" shortcut must become REAL Rust returns before `if` is
correct (a diverging `return` branch unifies with the other
branch's DenseArray type).

## Steps
1. if-real-return -- lower_body emits REAL `return #v;` for every
   return statement (tail only for a non-return final expression);
   lower `if cond { then } else { else }` as a Rust if-expression
   over DenseArray truthiness (data()[0] != 0), branches via the
   shared lower_body. Update the compiler-functions return tests.
   while/for/records/Results still rejected. e2e: an if program
   compiles + runs.
2. while-for-mut -- mutable variables (first Assign -> `let mut`,
   later Assign -> reassign) + lower `while`/`for`. (Accumulator
   loops need mutation.)
3. records-results -- RecordLit + FieldAccess + Results (ok/err/?)
   in the compiled value model.
4. close -- queue + companion-demo-file-processing mark; --done.
