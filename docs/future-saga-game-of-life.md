# Future saga: Game of Life + the animated grid widget

Seed input for `agentrail init`. Answers three questions asked on
2026-07-26: how far is sw-MLPL from the APL2 Game-of-Life one-liner,
can we ship a Life demo, and what animatable grid widget should carry
it.

## The APL one-liner vs sw-MLPL today

The classic APL expression

```text
life <- {take 1 omega or.and 3 4 = +/ , -1 0 1 outer.rotate1
         -1 0 1 outer.rotate2 enclose omega}
```

(using ASCII names for take/mix, or.and, outer-product, the two
rotates, and enclose) leans on five APL primitives. Where each stands
in sw-MLPL, mapped against `docs/apl2-staging-plan.md`:

| APL primitive | Role in the one-liner | sw-MLPL today | Parity plan |
| --- | --- | --- | --- |
| rotate (both axes) | shift the board 8 ways | MISSING as a builtin; expressible as matmul by a cyclic permutation matrix | Stage 4 (`rotate`, `reverse`) |
| outer product | build the 3x3 stack of shifted boards | only `matmul`; no general `outer(:f, a, b)` | Stage 3 |
| generalized inner product (`or.and`) | apply the birth/survive rule | only `matmul`; no `inner(:f, :g, a, b)` | Stage 3 |
| ravel + reduce (`+/,`) | count neighbors across the stack | `reduce_add` exists; general `ravel` is Stage 4 | Stage 4 |
| enclose / mix | treat the whole board as one cell | nested arrays do not exist | Stage 6 (the data-model stage) |

So a FAITHFUL transliteration needs Stages 3 + 4 + 6. But a working,
idiomatic-array Life needs NONE of them -- verified live on
mlpl-serve (2026-07-26): cyclic shift is a matmul with a permutation
matrix, and the whole engine is three lines:

```text
R = one_hot(concat(iota(n - 1) + 1, [0], 0), n)   # shift-by-one permutation
def u:life(G) { u = matmul(R, G) ; d = matmul(transpose(R), G) ;
  l = matmul(G, transpose(R)) ; r = matmul(G, R) ;
  N = u + d + l + r + matmul(R, l) + matmul(R, r)
    + matmul(transpose(R), l) + matmul(transpose(R), r) ;
  gt(eq(N, 3) + G * eq(N, 2), 0) }                 # B3/S23
repeat k { G = u:life(G) }
```

Glider test: after 4 generations the glider translated exactly one
cell diagonally with population preserved (5) -- the canonical
correctness check. Toroidal wrap falls out of the cyclic permutation
for free, matching APL's rotate semantics.

## The functional shape (preferred)

`docs/life-apl2-rust-research.txt` (keyword-APL + Rust-iterator study)
lands on the key unrolling: the one-liner's `or.and` inner product IS
the pure predicate `(N == 3) OR (N == 4 AND cell)` over the
SELF-INCLUSIVE 9-neighborhood count -- no self-exclusion special case,
and the rule reads like the rulebook. Verified live (2026-07-26) as a
functional pipeline structurally mirroring the APL
outer-product-of-shifts, and byte-identical to the 8-neighbor variant
across 4 glider generations:

```text
def u:shift(G, dr, dc) { ... matmul(Rr, matmul(G, transpose(Rc))) }
def u:life(G) {
  N = fill([n, n], 0) ;
  for dr in [0 - 1, 0, 1] { for dc in [0 - 1, 0, 1] {
    N = N + u:shift(G, dr, dc) } } ;
  gt(eq(N, 3) + G * eq(N, 4), 0)     # (N==3) or (N==4 and alive)
}
```

Trade-offs on the user's axis (readability vs allocations /
conciseness / speed / memory):

| Approach | Readability | Cost profile |
| --- | --- | --- |
| Array style, permutation matmuls (today) | medium ("rotation IS a matmul" is on-voice, but Rr/Rc selection is noise) | 9 shifted boards per step; each shift O(n^3) matmul -- trivial at n<=32 |
| Array style with Stage-4 `rotate` | high -- `rotate(rotate(G, dr, 0), dc, 1)` reads like the APL | same 9 temporaries, shifts drop to O(n^2); the sweet spot |
| With Stage-3 `outer`/`inner` too | a genuine transliteration of the one-liner | as above; conciseness maxed |
| Rust lazy-iterator style (per-cell pull, zero intermediate boards) | n/a at the language level | that is ENGINE territory: the natural lowering target if `rotate`-based Life ever goes through the compile-to-Rust path (`mlpl-build`), or a native stencil op -- noted, not needed at demo scale |

The demo ships the functional 9-neighborhood form (with `rotate` once
step 1 lands); the APL2 strengths kept are whole-array shifts, the
boolean-mask rule, and zero explicit cell loops. The per-cell lazy
style is deliberately rejected at the language level -- it trades away
exactly the array-thinking the platform teaches.

**What is missing is therefore not capability but LEGIBILITY**: with
Stage 4's `rotate(x, k, axis)` the 8 shifts collapse to
`rotate(rotate(G, dr, 0), dc, 1)` and the demo reads like the APL
original; with Stage 3's `outer`/`inner` it becomes a genuine
transliteration. Life is the ideal motivating demo for both stages.

## The animated grid widget: `svg(frames, "life")`

Today each generation renders as a separate `svg(G, "heatmap")` --
static frames stacked down the page. The proposal: one new mlpl-viz
render mode taking a rank-3 `[T, H, W]` tensor and emitting a single
self-contained SVG that ANIMATES through the T frames using SMIL
(`<animate>` on per-frame group opacity, staggered `begin` times,
`repeatCount="indefinite"`).

Why SMIL instead of a live Yew widget:

- Pure string assembly in mlpl-viz -- native-testable like every other
  chart, no new UI crate, no JS.
- Works everywhere the existing SVG pipeline works: inline results,
  the persisted-entry path, the PUBLIC pages demo, even the SVG
  download button (the downloaded file still animates).
- Browsers (Chrome/Firefox/Safari) all play SMIL; no script, CSP-safe.

Building the frames tensor needs no new language features either:
`F = fill([1, n, n], 0)`-style accumulation via
`concat(F, reshape(G, [1, n, n]), 0)` inside the `repeat` loop.

## Saga steps

1. **rotate-builtin** -- Stage 4 down-payment: `rotate(x, k, axis)`
   (cyclic, negative k allowed), autograd pass-through (a pure
   permutation), tests vs the permutation-matmul construction.
   Small, and instantly makes Life + future Stage 4 demos readable.
2. **life-widget** -- `svg(frames, "life")` SMIL grid animation in the
   viz crates (post tech-debt split there is room); golden test on a
   2-frame blinker (frame groups + animate timing in the output);
   render fallback: T=1 behaves like heatmap.
3. **life-demo** -- "Game of Life (APL classic)" demo, Classical ML
   ... no: its own "Array Classics" or the existing Visualizations
   category; glider + blinker, `rotate`-based step fn, ~24-frame
   animation, intro tying the APL one-liner to the staging plan,
   glossary entry [[Game of Life]] + [[Rotate]] cross-links, README
   counts, pages rebuild. Eval test: glider translation invariant
   (the live-verified assertion).

## Notes

- Engine verified with zero new builtins; steps 1-2 are what turn it
  from possible into beautiful.
- Board sizes stay small (8..32); n=32, 24 frames is trivial CPU work
  and browser-tier.
- The permutation-matrix trick deserves a line in the demo either way
  -- "rotation IS a matmul" is exactly this platform's voice.
