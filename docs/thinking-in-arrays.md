# Thinking in Arrays: loops you keep, loops you lose

Companion essay to the "Thinking in Arrays (loops you keep,
loops you lose)" demo in the web playground's APL2 / General
Programming category. The demo shows the three cases in 15
lines; this doc explains WHY the line falls where it does, and
why the same line runs through the middle of modern ML.

The APL tradition sw-MLPL inherits is often summarized as
"avoid explicit loops." That summary is half right. Array
notation eliminates one KIND of loop completely and cannot
eliminate the other kind at all -- and knowing which is which
is the actual skill. The distinction is about where the
dependency between iterations lies.

## Loops over data: iterations are independent

Consider the loop a scalar-language programmer writes to total
an order list:

```text
total = 0
for i in [0, 1, 2, 3, 4, 5] {
    total = total + take(prices, 0, i) * take(qty, 0, i)
}
```

Iteration 3 never reads anything iteration 2 wrote (the running
`total` is an accumulation artifact of the loop FORM, not of the
problem). Every iteration reads its own element and contributes
one independent product. Because the iterations are independent,
their order is arbitrary -- they could run backwards, shuffled,
or all at once. That freedom is exactly what lets the loop
vanish into a primitive:

```text
reduce_add(prices * qty)
```

The elementwise `*` performs every per-element iteration
simultaneously; the reduction folds them. The loop index was
just an ADDRESS into data, and array languages fold addressing
into the operations themselves. This is the classic APL move --
and it is also why the batch dimension in ML is "free": one
matmul applies a layer to every example in the batch, one
softmax normalizes every row. No cell loop survives a Game of
Life generation for the same reason: eight `rotate`s and a `+`
count every cell's neighbors at once.

## Loops over time: each iteration consumes the last one's output

Now the second loop from the demo:

```text
bal = 100
repeat 10 { bal = bal * 1.05 }
```

Written as what it is:

```text
bal(t+1) = bal(t) * 1.05
```

This is a RECURRENCE. Year 7's balance is an input to year 8's
computation; the iterations are chained, not independent. The
index is not addressing data -- it is advancing a STATE through
a trajectory, which is why we call it a loop over time: `bal(t)`
genuinely is "the balance at time t." No array primitive, in APL
or anywhere else, can remove a loop whose iterations feed each
other. `repeat`-until-exit is even more clearly temporal: the
exit test reads the state accumulated so far, so the iteration
count is not even knowable in advance.

Training is the canonical case:

```text
train 20 { adam(cross_entropy(apply(m, X), y), m, ...) }
```

is the recurrence `w(k+1) = f(w(k))` -- step k's gradients are
evaluated AT the weights step k-1 produced. Twenty steps means
twenty sequential passes regardless of notation. What array
thinking still buys you is the BODY: the entire forward pass,
loss, backward pass, and update inside one step is a single
line of array expressions.

Game of Life exhibits both kinds in one program, which is why
the playground's Life demo is the best teaching example: one
GENERATION is loop-free (a data computation over the grid), but
STEPPING generations is `grid(t+1) = u:life(grid(t))` -- a
recurrence that stays an explicit `repeat` even in APL.

## The borderline: associative recurrences become scans

The line between the two kinds is not quite "time is never
vectorizable." A running product is also a recurrence --
`p(t+1) = p(t) * x(t)` -- yet the demo computes the entire
10-year balance trajectory in one expression with the
running-product builtin, `running_product`. The escape hatch is
ASSOCIATIVITY: because `(a*b)*c = a*(b*c)`, the sequential
chain can be re-grouped into a balanced tree and evaluated
largely in parallel. The dependency was only apparent, an
artifact of writing the chain left-to-right.

APL knew this: scan (`+\`, `x\`) is a first-class operator, and
the general `scan(:op, a[, axis])` form is planned for sw-MLPL
in the APL2-parity track (docs/apl2-parity-gap.md, gap G1).
Computer science knows it as the prefix-sum / parallel-scan
family (Blelloch's work-efficient scan is the standard
construction). The test is simple:

- Is the recurrence built from an associative op with no other
  dependence on `t`? Then it is a scan -- the time loop is
  absorbable.
- Is it nonlinear or non-associative in the state -- like
  `w - lr * grad(L, w)` -- then re-grouping is illegal and the
  sequence is irreducible.

This borderline is live research territory in ML: state-space
models (the Mamba family) get their speed precisely by forcing
their recurrences into associative form so a parallel scan can
evaluate what looks like a sequential RNN.

## Why this matters for sw-MLPL's roadmap

The project's ML surface lives on both sides of the line:

- DATA side: batches, attention (every query/key pair scored in
  one matmul -- the double position loop never exists),
  reductions, normalizations. Array notation already owns these.
- TIME side: optimizer steps (`train N`) and autoregressive
  generation (each token conditions on the ones before it).
  These cannot be notation-ed away; they can only be ENGINEERED
  faster. That is exactly the generation-speed track in
  docs/future-sagas-queue.md: a KV cache removes redundant
  data-side work from inside the time loop, and MTP
  self-speculation loosens the time dependency itself by
  guessing several steps ahead and verifying the guesses in
  parallel -- with an exact verifier, so semantics are
  unchanged.
- BORDERLINE: `scan` lands with the APL2 higher-order-function
  saga, carrying this pedagogy with it.

## The one-sentence version

Array notation eliminates loops whose index walks over the data
inside one state, keeps loops whose index walks the state
forward through time, and absorbs exactly those time loops
whose recurrence is associative -- everything else about "loop
avoidance" is a corollary.

## See also

- The "Thinking in Arrays", "Game of Life (APL classic)", and
  attention demos in the web playground.
- docs/apl2-parity-gap.md (gap G1: scan and the operator
  algebra).
- docs/future-sagas-queue.md (Track 1: the generation-speed
  program).
- docs/benchmarks.md (what the E4 resident tape did to the cost
  of each time-loop step on MLX).
