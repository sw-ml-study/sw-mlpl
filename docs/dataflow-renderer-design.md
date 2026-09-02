# `dataflow(nodes, edges)` -- a structural SVG renderer

Design for the one genuinely new visualization capability the SPM
curriculum (../emufpga `docs/research2.txt` section 10) and general
architecture teaching need: a renderer for **structure**, not
quantity. Every existing `svg()` mode -- scatter, line, bar, heatmap,
decision boundary, waffle -- answers "how much"; none draws "what flows
into what". `dataflow` fills that gap: boxes, directed edges, groups,
edge labels, optional widths, optional highlight.

Same facility explains a transformer block, an autograd graph, a
compiler-pass pipeline, an ML pipeline, MoE routing, and the SPM
`storage -> FIFO -> lanes -> accumulators` occupancy story. That
generality is why it belongs in sw-mlpl, not in one demo repo.

## Non-goals

- **Not Graphviz.** No spline routing, no port constraints, no
  general force-directed layout. `dataflow` targets **layered DAGs**
  (dataflow moves one direction), which is exactly the shape of the
  teaching diagrams above.
- **Not quantitative.** Node size encodes nothing by default; edge
  width is an OPTIONAL channel the caller drives, not an axis.
  `width_scale: "linear"` (default) treats the values as visual widths
  and clamps them into a display band so a raw quantity can never
  become a monstrous stroke; `width_scale: "log"` normalizes them by
  natural log, so an extreme ratio (269 MB vs 4 KiB, 65,000:1) reads as
  an honest orders-of-magnitude contrast on one picture instead of one
  edge blowing out the canvas and the other vanishing.
- **Recurrence via back-edges.** A DFS marks any edge into an ancestor
  (a recurrence, e.g. a rotating parameter store) as a back-edge. The
  ranking pass ignores back-edges so layering stays a strict DAG, and
  they render dashed, routed through a reserved bottom lane as a
  "rewind" (drop out of the source, run back, rise into the target) so
  the loop reads as going backward. A self-loop counts as a back-edge.

## Surface

A dedicated builtin, NOT an `svg()` mode: `svg(data, type)` dispatches
on a numeric `DenseArray`, and node/edge structure does not fit that
shape. `dataflow` takes two RECORDS and returns an SVG string, exactly
like `svg` returns a string:

```
positions = dataflow(nodes, edges)   # -> SVG string; disp/print or svg-embed it
```

The two arguments are **columnar records** (arrays/lists per field --
the MLPL-idiomatic encoding, and the shape every other builtin already
produces):

### `nodes`

| field | type | required | meaning |
| --- | --- | --- | --- |
| `labels` | `StrList` | yes | one box label per node; index = node id |
| `groups` | `Array` (int) | no | group id per node (same id -> boxed together) |
| `highlight` | `Array` (0/1) | no | 1 = draw the node in the highlight style |
| `col_gap` | number | no | override the minimum column gap in px (auto-widening for long labels still applies as a floor) |
| `row_gap` | number | no | override the row gap in px |
| `ranks` | `Array` (int) | no | explicit layer override (else computed) |

### `edges`

| field | type | required | meaning |
| --- | --- | --- | --- |
| `from` | `Array` (int) | yes | source node id per edge |
| `to` | `Array` (int) | yes | target node id per edge |
| `labels` | `StrList` | no | edge label per edge (drawn at midpoint) |
| `widths` | `Array` (num) | no | stroke width per edge (default 1) |
| `width_scale` | `Str` | no | `"linear"` (default, clamped) or `"log"` (normalize extreme ratios) |
| `highlight` | `Array` (0/1) | no | 1 = draw the edge in the highlight style |

`from` / `to` / and every optional edge array must share the edge
count; the node arrays must share the node count. Mismatches are a
clean `VizError`, never a panic. Ids out of range are a `VizError`.

Example -- the SPM pipeline:

```
nodes = {
  labels: ["storage", "FIFO", "lanes", "accumulators"],
  groups: [0, 1, 1, 2]
}
edges = {
  from: [0, 1, 2],
  to:   [1, 2, 3],
  labels: ["stream", "issue", "reduce"],
  widths: [3, 2, 1]
}
dataflow(nodes, edges)
```

Group 0 (memory) and group 2 (result) frame the store-bound vs
compute-bound contrast; the widths tell the occupancy story.

## Layout

A minimal layered (Sugiyama-lite) pass -- three deterministic steps,
no iteration to convergence:

1. **Rank (assign layers).** If `nodes.ranks` is given, use it.
   Otherwise longest-path from the sources: `rank(v) = 0` for a node
   with no incoming edge, else `1 + max(rank(u))` over predecessors
   `u`. One left-to-right topological sweep (Kahn's algorithm order);
   a detected cycle drops its back-edges from the ranking pass and
   marks them for dashed rendering.
2. **Order within a layer.** Group nodes of the same rank into a
   column. Order them by the barycenter of their predecessors' y
   positions (one pass, top rank downward) to reduce crossings; ties
   keep input order. Good enough for teaching DAGs, cheap, stable.
3. **Assign coordinates.** Column x by rank (`rank * COL_W`), node y by
   in-column position (`pos * ROW_H`), each centered in its band.
   Canvas size derives from max rank and max column height.

Layers flow **left to right** by default (the reading direction for
"only ever moves forward"); a `direction` field (`"lr"` / `"tb"`) is a
later option.

## Rendering

Plain SVG, self-contained, theme-matched to the playground (the same
palette `svg()` marks use), emitted as a string:

- **Node box**: `<rect rx=6>` + centered `<text>`; highlighted nodes
  swap fill/stroke tokens. Box width fits the label (monospace metric
  estimate), height fixed.
- **Group band**: for each `groups` id, a rounded `<rect>` behind that
  group's node bounding box, a faint fill, and a small corner label.
  Drawn first (under the nodes).
- **Directed edge**: a polyline from the source box's right edge to the
  target box's left edge (elbow through the mid-x), ending in an
  arrowhead via one shared fixed-size `<marker>` def (`markerUnits`
  `userSpaceOnUse`, so a wide edge thickens the line, not the
  arrowhead). `stroke-width` = the edge's `widths` entry; highlighted
  edges swap the stroke token; back-edges are dashed.
- **Edge label**: `<text>` on a small background plate, anchored at the
  center of the first column gap after the source -- always clear of the
  node boxes, so even a skip-edge label (whose edge routes past an
  intermediate column) never lands on a box. The gap widens to fit the
  widest label it holds.

All geometry is integer-friendly and pre-computed in layout, so the
render step is a straight data -> string map (no measurement round
trips) -- keeps each function within the 25-LOC budget.

## Where it lives

A new leaf crate `components/viz/crates/mlpl-viz-flow`, mirroring
`mlpl-viz-marks` / `mlpl-viz-analysis`, with four small modules
(docs/code_metrics.md file-naming):

- `model.rs` -- the typed `Nodes` / `Edges` / `Positioned` structs.
- `parse.rs` -- records -> typed model (`Record` field extraction +
  validation -> `VizError`). This is the only surface that knows the
  MLPL record schema.
- `layout.rs` -- model -> positioned (rank, order, coordinates).
- `render.rs` -- positioned -> SVG string (boxes, groups, edges,
  labels).

`mlpl-viz` re-exports `render_dataflow(nodes, edges)`. It is NOT a
`render_with_aux` type arm, because its inputs are records, not a
`DenseArray`; it is a sibling entry point.

**Interpreter builtin.** `dataflow` is intercepted in `mlpl-eval`
(alongside `svg` in `eval_ops` / a new `eval_dataflow`): evaluate the
two arguments to `Value::Record`, hand them to
`mlpl_viz::render_dataflow`, return `Value::Str(svg)`. Web/`-f`/REPL
only, exactly like `svg` -- it is a visualization surface, so it does
NOT lower on the compile-to-Rust path (the compiler has no viz stack).

## Errors

All via `VizError` (surfaced as an `EvalError` at the boundary, like
`svg`):

- node/edge column length mismatch (`from`/`to`/optional arrays);
- an id in `from`/`to` outside `0..node_count`;
- a `groups` / `highlight` length that does not match the node/edge
  count;
- empty `labels` (a dataflow with no nodes).

## Phasing

- **Phase 1 (MVP)**: `labels` + `from`/`to`, longest-path layering,
  left-to-right, boxes + arrowheads + edge labels. Proves the seam
  end to end with the SPM chain and a transformer block. Ships with
  its demo (per the Viz-IR "no renderer without its proof demo" rule,
  docs/viz-ir-plan.md).
- **Phase 2**: `groups` bands, `widths`, `highlight` -- the SPM
  occupancy + memory-hierarchy contrasts and the monotonic-traversal
  highlight.
- **Phase 3**: dashed back-edges for the recurrence case (shipped);
  barycenter crossing reduction and a `direction` option remain.
- **Future**: fold under the Viz IR (research2 section 11) so a model
  graph, an autograd tape, and a compiler pass all lower to the SAME
  `dataflow` node/edge IR rather than each hand-building the records.

## Worked examples

Transformer block (structure, not numbers):

```
nodes = { labels: ["x", "rms_norm", "attn", "+", "rms_norm", "mlp", "+"],
          groups: [0, 1, 1, 0, 2, 2, 0] }
edges = { from: [0, 1, 2, 0, 3, 4, 5, 3],
          to:   [1, 2, 3, 3, 4, 5, 6, 6],
          labels: ["", "", "", "residual", "", "", "", "residual"] }
dataflow(nodes, edges)
```

The two `residual` edges skip a rank -- the layered layout draws them
as the long forward arrows that make a residual connection legible,
which is exactly what a line chart cannot show.

## Testing

- `mlpl-viz-flow` unit tests: layout ranks for a chain, a diamond, and
  a residual skip; `parse` rejects each error shape; `render` output
  contains the expected `<rect>` / `<marker>` / `<text>` counts.
- An `mlpl-eval` integration test: `dataflow({labels:[...]}, {from:[...],
  to:[...]})` returns a non-empty `<svg ...>` string.
- A `Basics`/`Visualization` web demo (the SPM pipeline) as the proof
  demo, wired through `svg`-style embedding in the playground.
