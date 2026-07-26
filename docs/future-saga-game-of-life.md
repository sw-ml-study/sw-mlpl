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

## APL2-ness audit: what the research kept and what it dropped

The keyword-APL study (and both verified variants above) solve Life
with the APL SUBSET -- flat arrays, shifts, reductions, masks. The
one-liner's genuinely APL2 parts got unrolled away:

| One-liner fragment | APL2 feature | Status in the unrolled forms |
| --- | --- | --- |
| enclose omega / take-mix | nested arrays (THE APL2 feature) | dropped -- board stays flat |
| the strand `1 omega` | heterogeneous 2-item vector (scalar + whole board as items) | dropped -- folded into the rule predicate |
| outer-rotate over the boxed board | operators mapping over nested items | approximated by loops / a rank-3 stack |
| or.and against the strand | generalized inner product over nested operands | unrolled to `(N==3) or (N==4 and alive)` |
| rotate, `+/`, `=` | classic APL, NOT APL2-specific | these are the parts we kept |

A third variant, verified live 2026-07-26, is the closest FLAT shadow
of the APL2 idiom: build the shifted boards as a rank-3 `[9, n, n]`
stack (offsets stored as data -- `dr`/`dc` lookup vectors, arrays as
control flow) and take `reduce_add(S, 0)` as the literal `+/` over the
stack axis. Identical glider evolution to both other forms. Two
findings along the way: `concat` grows cleanly from an empty
`fill([0, n, n], 0)` tensor, and `for` requires loop-carried values to
keep a constant shape (grow with `while` instead) -- a limitation
Stage 3's `outer` dissolves, since APL2 would not loop at all.

## APL2 opportunities (general, ranked by ML value)

1. **Stage 6 nested arrays** -- ragged token sequences and
   lists-of-tensors are the ML case that flat arrays genuinely cannot
   express; Stage 1's `depth`/`disp` are already waiting for depth to
   become real.
2. **Function operands for operators, including `u:` functions** --
   the `:op` BuiltinRef mechanism exists (reduce template); letting
   `each`/`outer`/`inner`/`scan` take USER functions is the
   generalized-operator spirit that made APL2 composable.
3. **Stages 2-3 (`each`/`cells`/`scan`, `outer`/`inner`)** -- planned;
   Life hands `outer`/`inner` their motivating demo before attention
   does.
4. **Indexed/selective assignment** -- APL2's selective assignment has
   no MLPL analog (no scatter). Life seeding needed a 64-element
   literal; kNN needed one_hot-matmul gathers. A `put(x, idx, v)` +
   vector-index gather pair is small and high-leverage.
5. **Strands / mixed vectors** -- mostly falls out of Stage 6 boxes;
   `Value::StrList` / records are the current stand-ins.

## Life as the APL2 staging showcase

- **v0 (today)**: functional 9-neighborhood, permutation-matmul
  shifts -- ships as the demo now.
- **v1 (today)**: the `[9, n, n]` stack + `reduce_add(S, 0)` -- the
  flat shadow of `+/,`; worth a demo line to foreshadow the nested
  version.
- **v2 (Stages 3-4)**: `outer(:shift, ...)`-built stack + `rotate` --
  reads like the one-liner minus the boxes.
- **v3 (Stage 6 capstone)**: enclose the board, outer-rotate the BOX
  into a 3x3 nested array of boards (`disp` renders a board-of-boards
  frame!), strand `[1, G]`, `inner(:or, :and, ...)` -- the APL2
  finale, and the natural acceptance demo for Stage 6 itself.

## The stencil formulation (fourth family)

A reader-supplied variant (2026-07-26, glyphs approximated):

```text
lw: 2 3 member stencil(3 3) rank(inner_or_and ravel-each (4 /= iota 9)) omega
```

This is the WINDOWED family: instead of shifting the whole board,
apply a function to each cell's 3x3 neighborhood. The pieces map to
the staging plan almost one-to-one:

| Fragment | Feature | sw-MLPL status |
| --- | --- | --- |
| rank operator (apply per window/subarray) | Stage 2 `cells(:f, k, x)` | planned |
| ravel-each + inner product with a mask | Stage 2 `each` + Stage 3 `inner` | planned |
| `4 /= iota 9` (drop the window center) | mask arithmetic | works TODAY: `1 - eq(iota(9), 4)` |
| `2 3 member` (count in the survive set) | Stage 5 `member(x, y)` | planned |
| the 3x3 sliding window itself | stencil / windowing | NOT in the staging plan |

The missing primitive -- sliding windows -- is not an APL2 gap to
patch separately: **a stencil IS a convolution.** Neighbor counting is
exactly `conv2d(G, K)` with the 3x3 all-ones-except-center kernel,
and `conv2d` is already step 2 of the parked stable-diffusion saga.
So the stencil Life lands for free when conv2d does, and becomes the
perfect bridge demo: "convolution is a sliding stencil; Life is its
hello-world" -- classical arrays shaking hands with CNNs, which is
this platform's whole voice. Recommendation: when the stable-
diffusion saga resumes, its conv2d step should cite Life as its
second acceptance demo (after the U-Net requirement).

## disp-driven narration (per user direction)

Every Life demo variant should narrate its DATA STRUCTURES with the
Stage-1 introspection builtins as it goes:

- `disp(G)` on the seeded board -- the glider as a framed grid;
- `size(S)` / `tally(S)` / `disp` on the `[9, n, n]` shifted stack --
  SEE the nine boards before they collapse into the count;
- `disp(N)` on the neighbor-count board -- the rule's input made
  visible (values 0..9, the 3s and 4s about to matter);
- in the Stage-6 version, `disp` of the enclosed 3x3 board-of-boards
  is the money shot: nested frames of little grids, APL2 depth made
  visible for the first time.

## Widget I/O: how sw-MLPL talks to a 2D display

Traditional APL used shared variables for device I/O. sw-MLPL has
three existing idioms, and the Life widget uses the first now with a
path to the third:

1. **Value return (pure)** -- `svg(...)` returns a string; the render
   pipeline displays it. No channel at all: display is a VALUE. The
   `svg(frames, "life")` SMIL widget lives here -- the whole animation
   is one self-contained value, replayable, downloadable, and correct
   on the static pages site. This is the functional answer and the
   default.
2. **Event emission** -- the 3D stage: eval emits `Stage3dEvent`s
   through `mlpl_web_viz3d::events` to a JS hook. One-way fire-and-
   forget; closest in spirit to an APL shared variable's write side.
3. **Generation-keyed trace stores + streaming** -- the live-loss
   pipeline: the eval side PUSHES into a named per-eval mailbox
   (`loss_trace`, `telemetry_trace`), the widget POLLS it on its own
   clock, SSE carries it across the network. This IS the shared-
   variable pattern, modernized: a rendezvous cell both sides touch
   without knowing each other. A future LIVE Life (watch generations
   on the server as they compute, not replayed) would add a
   `frame_trace` store fed by a tensor-frame SSE event alongside
   `metric` -- same architecture as the live loss panel, one payload
   type bigger. UPGRADE (user direction, 2026-07-25): the live
   `frame_trace` finale is now IN SCOPE for the Life saga -- the demo
   should walk the supported approaches and END on the best one shown
   iterating LIVE, not only as an SMIL replay. See step 4 below.

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
3. **life-demo** -- "Game of Life (APL classic)" demo (its own
   "Array Classics" or the existing Visualizations category), shaped
   as a MULTI-VARIANT showcase (user direction): walk the three
   verified engines in ascending APL2-ness -- (a) 8-neighbor
   permutation-matmul, (b) the functional 9-neighborhood
   `gt(eq(N,3) + G*eq(N,4), 0)` form, (c) the rank-3 `[9,n,n]`
   shifted-stack + `reduce_add(S,0)` form -- with `disp` narration of
   each intermediate structure (board, shifted stack, neighbor
   counts), then crown the best one (c, the closest to the APL
   one-liner's spirit) and run it for ~24 generations into the SMIL
   animation. Glider + blinker seeds (seeding = the lens `put` idiom,
   see docs/functional-lenses.md), intro tying the APL one-liner to
   the staging plan, glossary entries [[Game of Life]] + [[Rotate]],
   README counts, pages rebuild. Eval test: glider translation
   invariant (the live-verified assertion).
4. **life-live (frame_trace)** -- the finale runs CONNECT-side with
   live iteration display: a `frame` SSE event (shape + flat values
   per generation, mirroring `metric`), a generation-keyed
   `frame_trace` store (the `loss_trace` pattern), and a live grid
   panel that polls it (the `LiveLossPanel` pattern) -- APL shared
   variables, modernized. Demo's last act: `repeat 60` on the server
   with the grid animating as generations ARRIVE, then the SMIL
   value persists as the final entry (same live-then-persist shape
   as the loss curve).

## Notes

- Engine verified with zero new builtins; steps 1-2 are what turn it
  from possible into beautiful.
- Board sizes stay small (8..32); n=32, 24 frames is trivial CPU work
  and browser-tier.
- The permutation-matrix trick deserves a line in the demo either way
  -- "rotation IS a matmul" is exactly this platform's voice.
