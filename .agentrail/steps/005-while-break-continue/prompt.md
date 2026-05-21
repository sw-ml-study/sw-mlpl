Scripting saga step 005: add while + break + continue.

Surface forms:
  while cond { body }         -- loop until cond is non-truthy
  break                       -- exit innermost loop, loop value is the last body value
  break val                   -- exit innermost loop, loop value is val
  continue                    -- skip to next iteration

break and continue work inside any loop form: while, repeat, train, for. break outside a loop is a parse error or runtime error (parser-side check preferred).

The while loop's value is the last body value of the last iteration (matches repeat semantics). break value overrides that.

TDD:
- RED: parser tests for the surface forms (basic while; while with break; while with continue; break outside loop must error).
- RED: eval tests asserting:
  - while loop terminates when cond becomes falsy
  - break exits the loop and returns the last body value
  - break value returns value
  - continue skips to next iteration
  - break inside a nested while only exits the innermost loop
  - break inside repeat / train / for works the same way
- GREEN: extend the AST with Expr::While { cond, body }, Expr::Break(Option<Expr>), Expr::Continue. Implement eval rules. The eval loop probably needs a BreakSignal / ContinueSignal that propagates up to the enclosing loop's eval; design that carefully to not leak through non-loop scopes (e.g. a function call should NOT propagate break out of its caller's loop).

Quality gates: cargo test workspace; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md.

Glossary: add 'while loop', 'break', 'continue' entries to docs/glossary.md.

After this step ships scripts can write adaptive loops (e.g. 'train until validation loss stops dropping').