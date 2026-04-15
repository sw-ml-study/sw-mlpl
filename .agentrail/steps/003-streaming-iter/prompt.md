Phase 1 step 003: Streaming iteration over rows. Add a new language construct: `for <ident> in <expr> { <body> }`.

Semantics: expr evaluates to an array of rank r >= 1. The body runs once per row (once per rank-(r-1) slice along axis 0), binding `ident` to that slice in the environment. On each iteration, the body's final expression's value is captured into a per-iteration array and exposed in the environment as `last_rows` after the loop completes. Labels on axis 0 of the source are dropped; the slice inherits labels from axes 1..r.

Parser: add a Token::For keyword ("for"); add an Expr::For { binding: String, source: Box<Expr>, body: Vec<Expr>, span: Span } variant; the grammar production is `for IDENT in expr { body }`. Update describe_kind for the new token. Keep this narrow -- no C-style `for(i=0;i<n;i++)`, no else clause.

Evaluator: implement eval_for in eval.rs alongside eval_repeat/eval_train. Error clearly if the source is not rank >= 1.

Compile path: for now, Expr::For in mlpl-lower-rs returns LowerError::Unsupported. A future compile-path saga will lower it.

TDD:
(1) parse `for row in m { shape(row) }` into the new Expr::For variant.
(2) running `for row in reshape(iota(6), [3, 2]) { reduce_add(row) }` yields last_rows equal to [1, 5, 9].
(3) non-array source errors.
(4) existing repeat and train demos still pass.
(5) lower-rs rejects Expr::For as Unsupported.
