# Partial application and the combinator birds: design

Status: committed design (saga combinator-birds, step 001).
Source brief: docs/combinators-research.txt ("To Mock a
Mockingbird" program; target = the brief's Level 2). The
companion inspiration is the fluent combinators example at
mlajtos.github.io/fluent.

## The one idea

Today a `:u:name` reference is a VALUE but application is not:
`call(f, args...)` either saturates the function or errors.
Combinatory logic needs application itself to produce values --
`K x` IS a thing you can call later. The brief's answer, which
this design adopts verbatim, is a third callable kind that
stores DATA, not code:

```text
Partial { callable: :u:name, bound: [values...] }
```

No lambdas, no lexical closures, no captured environments:
references stay named and late-bound; a partial only remembers
which definition and which arguments so far.

## Applicative `call`

`call(f, a1..an)` where `f` is a `:u:` reference or a Partial
with `k` arguments still missing:

| supplied vs missing | result |
|---|---|
| fewer (`n < k`) | a NEW Partial with the arguments appended -- no execution |
| exact (`n == k`) | execute |
| more (`n > k`) | execute on the first `k`; the result must itself be callable, and the leftovers apply LEFT-ASSOCIATIVELY: `call(f, a, b, c)` == `call(call(call(f, a), b), c)` |

Zero-argument `call(f)` on an unsaturated callable returns it
unchanged (identity of application). Excess application onto a
non-callable result is a loud error naming the value kind that
could not be applied.

## v1 scope: user functions only

Partials form over `:u:` references and over other Partials.
BUILTIN references keep exact-arity semantics: builtin arity is
not a fixed fact (`reduce` takes 2 or 3, `concat` 2 or 3), so
under-application of `:mean` has no well-defined "missing
count". The tutoring error says so and names the fix: wrap the
builtin in a `u:` definition. This also keeps the aviary
honest -- Smullyan's birds are user definitions.

## Storage and surface ripple

`Partial` is the twelfth `Value` kind. It binds like every
other value (assignment routes it; `clear_binding` sweeps it;
the frame snapshot covers it so partials obey the same scope
hygiene as everything else), passes as a user-function
argument, sits in record fields, and returns from functions --
those four flows are exactly what staged bird application
exercises. Display forms:

- `repr` / display: `partial(:u:B, 1 of 3 bound)`
- `equal`: structural -- same callable AND `equal` bound
  arguments, position by position
- `:describe name`: the same line repr shows, plus the bound
  arguments' kinds

The serve/wasm exhaustive matches (`value_kind`, the R1 payload
guard) gain arms; connect mode reports kind `partial`.

## Where partials are ACCEPTED

Everything that consumes a function reference learns to take a
Partial: `call` (the core), user-function arguments, `each` /
`table` / `atop` / `over`, the Result combinators (`map_ok` /
`and_then` / `or_else`), and `bracket` hooks. One shared
"resolve callable" helper keeps the semantics identical
everywhere: a Partial invoked with `m` arguments behaves as its
underlying function invoked with `bound + m`.

## What this deliberately does NOT add

Per the brief: no anonymous lambdas, no lexical closures, no
heterogeneous function arrays (a record is the registry), no
special combinator syntax, no trains/tacit forms yet (queued as
a future idea, noted in the research). `arity(f)` introspection
is optional and deferred until a consumer needs it.

## The demo (step 3)

A "Combinators (the birds)" web demo in APL2 / General
Programming: define the aviary as ordinary `u:` functions (I,
K, T, M, B, C, W, S), show STAGED application (`k5 =
call(:u:K, 5)` then `call(k5, 99)`), show the left-associative
equivalence, build a derived bird from the SK basis -- the
brief's acceptance criterion -- touch self-application with M,
and close with the uniquely-MLPL finale: one Bluebird
composition applied unchanged to a scalar, a vector, and a
matrix (combinators compose computation; pervasion supplies
the data parallelism). The idioms document gains a matching
section. Readable-word wrappers (`compose`, `flip`,
`constant`) are follow-up candidates once the demo shows which
spellings earn their keep.

## Acceptance

- Staged application: `call(call(call(:u:B, f), g), x)` ==
  `call(:u:B, f, g, x)` == `f(g(x))`, pinned by tests.
- The SK-basis test derives I from S and K (`S K K x == x`).
- Mockingbird self-application: `call(:u:M, :u:I)` terminates
  and returns `:u:I`-applied-to-itself's value.
- Every storage flow (variable, record field, argument,
  return) round-trips a Partial.
