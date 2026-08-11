# Saga: compiler-functions

demo-file-processing's next gate after `include`: the compiler
rejects user functions (`def u:`). Lower them -- PARAM-ONLY first
slice (user direction): a user fn reads only its parameters +
body-locals; reading a global is a clear Unsupported error (defer
globals + control-flow-in-body + records/results to later rungs).

- fndef_lower.rs: lower `def u:name(params) { body }` to a nested
  Rust `fn user_name(p: DenseArray, ...) -> DenseArray { body }`;
  free-variable check (reject reads of non-param/non-local names);
  body lowering (locals via Assign, `return`, doc-string statement
  discarded, tail expression = return value).
- lib.rs lower_stmt: a FnDef stmt lowers to that fn item pushed
  into the block's bindings (Rust hoists nested fn items, so call
  order is free).
- fncall.rs: `u:name(args)` FnCall routes to `user_name(args)`.
- DenseArray value model (numeric/array fns like page_dot4);
  string params/returns + records/results are later.

## Steps
1. fndef-lower -- fndef_lower.rs + lower_stmt FnDef + u: call
   routing; e2e compile test (a user-fn program runs) + lower unit
   tests + a free-var-rejection test; keep gates green.
2. close -- queue + companion-demo-file-processing mark
   (compiler-functions param-only shipped; next rung control-flow /
   records-results); --done.
