# Saga: combinator-birds
User direction; design brief: docs/combinators-research.txt
(target = its Level 2). Partial application as a VALUE --
Value::Partial {callable, bound args} -- with applicative
call(): under-application returns a Partial, exact arity
executes, excess arguments apply left-associatively to the
returned callable. No lambdas, no closures: references stay
named, partials store data. v1 scope: partials over USER
functions (fixed arity); builtin references keep exact arity
(their arity is not fixed -- wrap in a u: fn). Introspection:
repr "partial(:u:B, 1 of 3 bound)", equal structural,
:describe. HOF quartet + combinators accept partials.
## Steps
1. partial-design -- docs/combinators-design.md distilling the
   research into the committed semantics.
2. partial-core -- the value kind (12th), env table + binding
   ripple (assign/clear/snapshot/serve arms), applicative call,
   invoke path, TDD: staged K/B application, left-assoc
   equivalence, mockingbird self-application, SK-basis
   construction (the research's acceptance criterion).
3. birds-demo -- Combinators web demo (aviary record, staged
   application, SK basis, the pervasive-array composition
   finale) + idioms section + docs rows.
4. close -- rebuilds/deploy, wiki, queue advance.
