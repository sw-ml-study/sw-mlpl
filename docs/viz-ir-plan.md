# Viz IR + ML Visualizations: Saga Plan

## Status

- **Driver**: in-session conversation 2026-05-29; supersedes the
  free-form sketch in `docs/viz-ir.md` and complements
  `docs/3d-introspect-dialog.md`.
- **Decided ordering**: IR scaffold first; attention heatmap renderer
  (against `demos/attention.mlpl`) before the tiny-LM Sankey
  (`demos/tiny_lm.mlpl`). Simpler-first, both demos go through this
  doc's IR.
- **Already shipped (v0)**: clickable yellow pointer on the 3D stage
  opens a centered modal showing the selected sculpture's headline,
  shape/rank/elements/memory, stats, and a sampled values dump. That
  modal is the dispatch surface every renderer in this plan plugs
  into.

## Architectural principles

1. **One IR, many renderers.** The dialog dispatches on
   `VizKind`; renderers don't know about each other. Adding viz N+1
   means adding one match arm + one renderer module.
2. **Hand the renderer pre-computed data, not raw tensors.** The IR
   carries shape + summary + values *plus* viz-specific payloads
   (e.g. `AttentionViz` already has the softmaxed matrix + token
   labels). Renderers don't run MLPL.
3. **Dependency budget**: prefer what's loaded. Plotly for Sankey /
   scatter / line; raw SVG for heatmaps + custom; **D3 only when a
   Plotly-shaped solution measurably fails the UX** (custom
   hover-trace, animated forward-pass scrubber).
4. **Demo-driven scope**. Every saga ships a vertical slice: one
   MLPL demo opens, the user clicks a specific sculpture, sees the
   target renderer. No renderer ships without its proof demo.
5. **Components**: each new viz layer is its own component workspace
   following the saga 79+ pattern (≤4 crates per component, lib.rs
   as facade).

## Information needed from each target demo

The IR has to round-trip enough state for both renderers; doing
this analysis up front keeps saga 1 from designing the IR for
only the easier demo.

### `demos/attention.mlpl`

- Pure tensor pipeline, no model object, no real tokens.
- Targets the **attention heatmap** renderer.
- Drives these IR requirements:
  - Rank-2 tensors `[N, N]` need to be recognizable as "this is
    attention weights" regardless of dtype. The MLPL line that
    produced it (`A = softmax(scores, 1)`) is the semantic
    signal; user-typed code may not include `attention(...)`.
    **IR carries a producer expression string** so renderers
    can heuristic-detect.
  - Tokens are integer indices, not strings. The heatmap
    renderer must accept `Option<Vec<String>>` token labels and
    fall back to integer indices.
  - No batch / head axes. The IR's `AttentionViz` slot allows
    rank-2 directly; doesn't require rank-3+.

### `demos/tiny_lm.mlpl`

- Real BPE tokens, chain model with `causal_attention`,
  multi-step training loop.
- Targets the **composite Sankey** renderer.
- Drives these IR requirements:
  - Composite objects (`chain(...)`, `residual(...)`) need a node
    list with producer/consumer edges. **IR carries a graph
    payload** (`SankeyViz { nodes, edges }`) computed by the
    chain's `apply` rather than at click-time.
  - Real token strings need to be threaded down from the
    tokenizer (`apply_tokenizer(tok, ...)` results). The
    `AttentionViz` payload built from `causal_attention` must
    plumb the tokenizer's vocab for label resolution.
  - Heatmap renderer also needs to handle rank-3 `[head, q, k]`
    and rank-4 `[batch, head, q, k]` -- so a head selector +
    batch slice control are in scope from saga 2, not deferred.

### What's in scope from saga 1's IR

- Tensor metadata: shape, rank, elements, summary, optional values.
- Axis labels (already partly there: see
  `mlpl-web-viz3d/events.rs::ShapeInfo`).
- Producer expression string.
- Optional `viz_hint: VizKind` tag (caller-provided override).
- Composite graph payload (nodes + edges; empty for leaf tensors).
- Attention payload (query tokens, key tokens, weights matrix,
  causal flag, layer/head indices).

### What's out of scope from saga 1

- Loss-curve, distribution, embedding-projection payloads. They
  get added as the matching renderers come online.
- Live re-evaluation. Renderers consume snapshot data only.
- DR projections inside the dialog (queued in
  `docs/3d-introspect-dialog.md` saga step `dr-projection`).

## The IR (concrete Rust shape)

Crate: `components/web-viz-ir/crates/mlpl-web-viz-ir/`.

```rust
// lib.rs facade re-exports each of these from sibling modules
// (node.rs / attention.rs / sankey.rs / payload.rs).

pub enum VizKind {
    /// Scalar / vector / matrix tensor with values + stats.
    Tensor,
    /// Rank-2/3/4 matrix that downstream code should treat as
    /// attention weights. Carries an `AttentionViz` payload.
    Attention,
    /// Composite object (chain, residual, model). Carries a
    /// `SankeyViz` payload.
    Composite,
    /// Loss / metric series. Future.
    Series,
    /// 2-D projection of high-dim points. Future.
    Projection,
    /// Catch-all hint with no extra payload.
    Other(&'static str),
}

pub struct AxisDim {
    pub size: usize,
    pub label: Option<String>,    // "batch", "head", "y", "x", "token"
}

pub struct VizNode {
    pub id: String,                            // unique within trace
    pub name: Option<String>,                  // user-typed var name
    pub label: String,                         // pretty title for dialog header
    pub kind: VizKind,
    pub shape: Vec<AxisDim>,
    pub elements: usize,
    pub producer: Option<String>,              // "softmax(scores, 1)"
    pub stats: Option<ArraySummary>,           // existing type from viz3d events
    pub values: Option<Vec<f64>>,              // sampled or full
    pub attention: Option<AttentionViz>,       // populated for Attention kind
    pub sankey: Option<SankeyViz>,             // populated for Composite kind
}

pub struct AttentionViz {
    pub query_tokens: Vec<Token>,
    pub key_tokens: Vec<Token>,
    pub weights: Vec<f32>,                     // row-major [layer? * head? * q * k]
    pub layout: AttentionLayout,
    pub causal: bool,
    pub layer: Option<usize>,
    pub head: Option<usize>,
}

pub enum Token {
    Str(String),
    Index(usize),
}

pub enum AttentionLayout {
    QK { q: usize, k: usize },                 // saga 2: attention.mlpl
    HeadQK { head: usize, q: usize, k: usize },// saga 3: per-head selector
    BatchHeadQK { batch: usize, head: usize, q: usize, k: usize }, // saga 3
}

pub struct SankeyViz {
    pub nodes: Vec<SankeyNode>,
    pub edges: Vec<SankeyEdge>,
}

pub struct SankeyNode {
    pub id: String,
    pub label: String,                          // "embed (V=280, d=32)"
    pub op_kind: String,                        // "embed" / "attention" / "linear" / "softmax"
}

pub struct SankeyEdge {
    pub from: String,
    pub to: String,
    pub width: f64,                             // numeric magnitude (elements, token count, |grad|)
    pub label: Option<String>,                  // shape string for hover
}
```

`ArraySummary` is reused from `mlpl-web-viz3d::events`. The IR
crate depends on `mlpl-web-viz3d` for that single struct -- a
small backward dep, justified by avoiding type duplication.

## Saga sequence

Each saga ends with: build green, sw-checklist clean,
`scripts/build-pages.sh` rebuilt, live-deploy pushed, one
mention in `docs/saga.md`.

### Saga A: `viz-ir-scaffold`

**Goal**: ship the IR types + the dispatch site, no visible UX
change.

Steps:
1. Create `components/web-viz-ir/` workspace with the crate
   above.
2. Add an `Option<VizNode>` field to `HistoryEntry` and to the
   3D `userData` payload. Default `None` for now.
3. In `stage3d.js::renderInspectorBody`, dispatch on
   `userData.viz?.kind`: an absent kind falls through to the
   existing text body (today's behavior).
4. Build pages, push, smoke test (no visible change but
   nothing should regress).

### Saga B: `viz-attention-heatmap-attention-demo`

**Goal**: open `demos/attention.mlpl`, click the `A` sculpture,
see a real heatmap.

Steps:
1. Wire the evaluator: when `softmax(_, _)` produces a rank-2
   tensor on a path that starts from `matmul(Q, transpose(K))`
   or `attention(...)` / `causal_attention(...)`, populate
   `userData.viz = VizNode { kind: Attention, attention: Some(...) }`.
   Heuristic via the producer expression captured at trace
   time.
2. Add `mlpl-web-viz-renderers` crate (sibling to viz-ir).
   First renderer: `attention_heatmap.rs` -- pure SVG, color
   scale matches the existing `svg(_, "heatmap")` builtin's
   palette for consistency.
3. Dispatch in the inspector body: if `kind == Attention`,
   render the SVG heatmap + token-axis labels (indices for
   this demo); otherwise fall through to the existing body.
4. Demo proof: run `attention.mlpl`, click the sculpture for
   `A`, expect to see a 6x6 grid where each row sums to 1.

### Saga C: `viz-attention-heatmap-tiny-lm`

**Goal**: same renderer, real tokens, real attention from a
real model.

Steps:
1. Plumb the BPE tokenizer's reverse vocab through the
   evaluator so `causal_attention`'s output sculpture carries
   token strings, not indices.
2. Extend `AttentionLayout` plumbing (per-head selector,
   batch slice). The renderer gains two `<select>` controls:
   "head 0..H-1" and "batch slot 0..B-1".
3. Demo proof: run `tiny_lm.mlpl` (skip training: edit the
   demo or use the existing post-train snapshot), click the
   attention sculpture in the transformer block, see the
   familiar Alammar-style heatmap with real BPE pieces on the
   axes.

### Saga D: `viz-sankey-composite-tiny-lm`

**Goal**: click the `model` sculpture (the chain itself),
see a Plotly Sankey of token flow through embed → norm →
attention → MLP → out.

Steps:
1. Extend evaluator: `chain(...)` and `residual(...)` produce
   a sculpture whose `userData.viz` carries a `SankeyViz`
   payload built by walking the chain's `apply` graph at eval
   time.
2. Edge widths: use the post-op tensor's element count as
   the magnitude (so the bottleneck in an autoencoder becomes
   visually obvious). Open question: should embed and unembed
   share a vocab-sized edge, or render as a fan? Decide during
   the saga.
3. Add `sankey_composite.rs` renderer. Load `plotly.js` -- it's
   already on the page for existing scatter / line views.
4. Demo proof: run `tiny_lm.mlpl`, click the chain sculpture,
   see ribbons.

### Saga E: `viz-derivation-steps-attention`

**Goal**: click an attention sculpture, see the math
derivation as a step list (`Q = X · Wq`, `scores = Q · Kᵀ /
sqrt(d)`, ...). Each line is clickable: clicking re-opens the
inspector on the intermediate tensor.

Steps:
1. Capture the op chain that produced the selected tensor as
   part of `VizNode.producer`. Store as a parsed AST list, not
   a string, so the renderer can render each step.
2. Use MathJax (load on first open, not at page boot, to keep
   the cold-start fast).
3. Renderer: `derivation_steps.rs`. Demo proof:
   `attention.mlpl`, click `A`, see four lines.

### Saga F: `viz-d3-attention-trace`

**Goal**: upgrade the attention heatmap with D3 for
hover-to-trace.

Steps:
1. Add D3 to the page bundle (target: `~75 KB minified`,
   compare against full D3 since we likely only need
   `d3-scale` + `d3-selection` + `d3-array`). Pull in as a
   first-party JS asset rather than CDN to match the existing
   Three.js pattern.
2. Re-implement the attention heatmap in D3: hovering a cell
   highlights the row + column tokens, hovering a token label
   highlights its full row / column.
3. Same demo proofs as Saga B/C; verify no regression.

### Saga G: `viz-forward-scrubber-tiny-lm`

**Goal**: forward-pass step animation. The Transformer
Explainer's "wow" feature.

Steps:
1. Trace mod: when the user clicks the chain sculpture, the
   inspector renders the Sankey + a horizontal scrubber.
   Dragging the scrubber to step `i` re-renders intermediate
   activations frozen at op `i`.
2. Capture intermediates at apply time (already partly
   available via `:trace` infrastructure).
3. Demo proof: `tiny_lm.mlpl`, click chain, drag scrubber, see
   per-op output cards update.

## Renderer library at a glance

| Saga | Renderer file | Library | Bundle adds |
|------|---------------|---------|-------------|
| A | -- | -- | 0 |
| B | `attention_heatmap.rs` | SVG | 0 |
| C | (extends B) | SVG | 0 |
| D | `sankey_composite.rs` | Plotly Sankey | 0 (Plotly already loaded) |
| E | `derivation_steps.rs` | MathJax (lazy) | ~40 KB lazy |
| F | (rewrite B) | D3 | ~75 KB |
| G | (extends D) | Plotly + custom JS | small |

## Open questions

1. **Trace export format**: the doc proposes `:viz json <path>`.
   Decide at saga A whether to extend the existing
   `:trace json` or branch. Recommend: extend, since trace
   already has the timing + step structure we want.
2. **WASM-side vs JS-side rendering**: today's modal renders in
   JS. Sagas B-E could render server-side (in Rust → SVG
   strings) which keeps logic in one language and gets us
   sw-checklist coverage, OR client-side in JS for flexibility.
   Recommend: WASM-side SVG strings for heatmap + derivation;
   JS-side for D3 + Plotly (where the libraries live).
3. **Color palette consistency**: the existing `svg(_, "heatmap")`
   builtin already has a palette. Use the same for the dialog
   heatmaps so the two views read as "same thing, bigger".

## Out of scope (intentionally)

- 3D close-up scene inside the dialog. Queued in
  `docs/3d-introspect-dialog.md`; can ship in parallel.
- Composite-of-composite drill-down (encoder of encoder etc.).
  Defer until at least saga D ships.
- Training-loop visualization (loss curve + gradient flow
  Sankey). Saga H+ once D + E land.
- Inline animation of softmax / matmul (the Transformer
  Explainer's lowest-level layer). Defer indefinitely.

## References

- `docs/viz-ir.md` -- the brainstorm this plan grew from.
- `docs/3d-introspect-dialog.md` -- the dialog this plugs into.
- [Transformer Explainer (poloclub)](https://poloclub.github.io/transformer-explainer/)
- [Visualizing seq2seq + attention (Jay Alammar)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
