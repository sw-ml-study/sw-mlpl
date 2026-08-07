# Saga: y-combinator
User direction 2026-08-07: relate the Y COMBINATOR (as in the
brand casual users have heard of) to the birds demo, with
worked examples. Key finding (verified in the repl): MLPL
needs NO new language feature -- partials already supply the
delay a strict-language fixed-point combinator requires. Two
levels: (1) self-application recursion (the mockingbird move,
step takes self); (2) def u:fix(f, v) { call(f, call(:u:fix,
f), v) } -- a clean step + fix ties the knot; the partial
call(:u:fix, f) is the delayed recursive reference. Both
verified: fact 5 = 120, fib 10 = 55.
## Steps
1. y-examples -- extend the Combinators demo with a
   fixed-point section (self-application -> fix -> factorial /
   fibonacci -> "Y Combinator the brand is named after this");
   idioms doc + glossary entry; a Rust test pinning fix();
   verify demo runs + smoke.
2. close -- rebuild/deploy, wiki, q-and-a, README pins.
