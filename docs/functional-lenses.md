# Functional Lenses in sw-MLPL

Status: v1 idiom shipped in the Structure Zoo demo (get/put on a
vector). Composition sugar and first-class lens values are future
work tied to the APL2 staging plan. Research source:
`docs/apl-lense-research.txt` (APL2 lens sketch).

## What a lens is

A lens is a pair of pure functions focused on one part of a larger
structure:

- `get : whole -> part` -- read the focused part.
- `put : whole, new_part -> whole'` -- build a NEW whole with only
  the focused part replaced. The original is untouched.

Lenses compose: an outer lens (row of a matrix) chained with an
inner lens (cell of a row) yields a lens for the nested part.
Composition runs `get` outside-in and `put` inside-out.

This is exactly the functional-programming answer to APL2's
*selective assignment* (`(2 3 rho w) <- x`, written in APL2 with
the rho glyph): APL2 mutates through a
selection; a lens returns a fresh value through the same selection.
sw-MLPL prefers the lens (no mutation), and can later add selective
assignment as sugar that lowers to a lens `put`.

## The idiom today (no new builtins needed)

`take` is GET; a `one_hot` mask update is PUT:

```text
def u:get_at(x, i) { take(x, 0, i) }
def u:put_at(x, i, val) {
  m = reshape(one_hot([i], tally(x)), [tally(x)]);
  x * (1 - m) + val * m
}

v = [10, 20, 30, 40]
u:get_at(v, 2)          # -> 30
w = u:put_at(v, 2, 99)  # [10, 20, 99, 40]; v unchanged
```

Why this shape:

- `take(x, axis, i)` is already the language's "focus one slice"
  primitive, and it is tape-differentiable -- a lens `get` that
  works inside `train { }`.
- The mask update `x * (1 - m) + val * m` is the standard
  array-language functional setter. It is pure, shape-preserving,
  and also tape-differentiable, so a lens `put` can sit in a loss
  expression (e.g. counterfactual edits: "what is the loss if only
  this weight changed?").
- `scatter(buf, idx, vals)` is the bulk/accumulating cousin: many
  cells at once, but ADDING into the buffer. A pure multi-cell
  `put` is `scatter` onto a mask-zeroed base.

## Composition on nested structure (matrix cell)

Without broadcasting, the row mask for a matrix is an outer
product via `matmul` (the same no-broadcast trick used elsewhere):

```text
def u:get_rc(M, r, c) { take(take(M, 0, r), 0, c) }       # get: outside-in
def u:put_rc(M, r, c, val) {
  row = u:put_at(take(M, 0, r), c, val);                  # put: inside-out
  rm = matmul(reshape(one_hot([r], tally(M)), [tally(M), 1]),
              fill([1, size(take(M, 0, 0))], 1));         # [R,C] row mask
  M * (1 - rm) + matmul(reshape(one_hot([r], tally(M)), [tally(M), 1]),
                        reshape(row, [1, size(row)]))
}
```

Readable? Barely -- which is the point: this is the motivating
pain for the next two APL2 staging items.

## Where this goes (ties to the APL2 staging plan)

1. **`put` builtin** (`put(x, axis, i, slice)` -- non-accumulating
   scatter): collapses the mask dance to one call; `take`/`put`
   become the canonical lens pair. Ranked opportunity #3 in the APL2-opportunities
   section of `docs/future-saga-game-of-life.md` (noted as a
   stage-adjacent gap in `docs/apl2-staging-plan.md`).
2. **First-class `u:` function values** (opportunity #2): once a
   `u:` name can be passed like `:add`, a lens becomes a real pair
   value and `u:compose(lens_a, lens_b)` is writable in MLPL
   itself, matching the APL2 research sketch (lens = 2-item vector
   of functions).
3. **Selective assignment sugar** (Stage 6 nested-array era):
   `v[2] = 99` as surface syntax lowering to the lens `put` --
   APL2 ergonomics, functional semantics.
4. **Game of Life**: pattern seeding (stamping a glider into a
   zero board) is a rectangular-region `put` -- the Life saga's
   board setup is the second acceptance demo for the `put`
   builtin (see `docs/future-saga-game-of-life.md`).

## Demo surface

The Structure Zoo demo (Basics) closes with the vector lens act:
`disp(v)` / `disp(w)` before-and-after makes the "new value, old
value untouched" point visually. When the `put` builtin lands, the
demo swaps the mask body for `put(...)` and gains the matrix
composition act.
