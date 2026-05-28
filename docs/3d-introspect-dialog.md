# 3D object introspect dialog (proposed saga)

Click an affordance ("?" or "v") on a 3D sculpture in the stage to
open a large pop-up dialog showing an interactive close-up with
clickable aspects and labels that reveal the underlying tensor's
shape, statistics, contents, and (for composite objects like
encoder/decoder constructs) a drill-down into individual layers
with optional dimensionality reduction views.

## Use cases

### 1. Single-tensor inspector

A user runs

```mlpl
h = matmul(W, x)
```

A scalar/vector/matrix/tensor sculpture appears on the 3D stage.
Today: clicking it triggers `:describe h` in the REPL. Proposed:
clicking a small "?" / "v" affordance overlaid on the sculpture
opens a modal dialog with:

- A larger 3D view of just that tensor (centered, free rotate /
  pan / zoom independent of the main stage).
- Clickable labels for each axis ("batch = 32", "channel = 3",
  "y = 64", "x = 64") that, when clicked, slice the view along
  that axis (animated transition).
- A stats panel: min, max, mean, std, sparsity, NaN count.
- A histogram or distribution sparkline.
- A "raw values" expandable card with a tabular dump (paged for
  large arrays).
- A "copy to clipboard" affordance for the data.

### 2. Composite-object drill-down

A user runs a chain demo:

```mlpl
mdl = chain(linear(2, 8), tanh_layer(), linear(8, 3))
out = apply(mdl, x_batch)
```

The 3D stage shows the model as a connected chain of layer
sculptures with the input on one side and output on the other.
Proposed: clicking the affordance on the COMPOSITE sculpture (or
on any layer in it) opens a dialog where:

- The default view shows the chain with all layers labeled
  (Linear 2->8, TanH, Linear 8->3).
- Clicking any layer drills down into THAT layer's tensor view
  (case 1 above), with a "back to model" affordance.
- For a hidden layer, an additional toggle: "project to 2D
  (UMAP / t-SNE / PCA)" which routes the layer's activations
  through the existing DR builtins and re-renders the dialog
  scene as a 2D scatter of the projected representation.
- A breadcrumb at the top of the dialog: `mdl > Linear(8->3)`
  shows current depth.

### 3. Encoder/decoder

Most compelling case. A user runs

```mlpl
enc = chain(linear(64*64, 32), relu_layer(), linear(32, 8))
dec = chain(linear(8, 32), relu_layer(), linear(32, 64*64))
ae = chain(enc, dec)
out = apply(ae, x)
```

The 3D stage shows the autoencoder as a composite. The dialog:

- Splits naturally into Encoder | Bottleneck | Decoder regions.
- Clicking the bottleneck (the 8-D representation) opens a DR
  view of bottleneck activations across the input batch:
  - Coloured by ground-truth label if available.
  - Hoverable points showing the input that produced each.
  - DR algorithm picker (UMAP / t-SNE / PCA / MDS).
- Clicking encoder or decoder shows the full chain (case 2).
- A small "reconstruct" affordance triggers
  `apply(ae, x_subset)` and shows input vs. output side-by-side.

## Affordance design

The "?" / "v" affordance must:

- Appear on hover over a 3D sculpture (so the stage isn't
  cluttered when not interacting).
- Be reachable via keyboard (arrow keys to step through
  sculptures, Enter to open).
- Be discoverable from the splash / tour (saga 34 added the
  guided tour; a new tour step can point at the affordance).

Visual options to evaluate:

1. A 2D HTML overlay (positioned in screen space, like the
   existing tooltip).
2. A 3D billboard sprite (lives in scene space, stays oriented
   to camera).
3. A subtle outline / pulse on hover with a hint tooltip
   "click to inspect" and no inline icon.

Recommend (1) for simplicity + accessibility; (2) feels better
visually but is harder to make keyboard-accessible.

## Dialog framing

- Modal (page-blocking) vs. side-pane (REPL keeps working)?
  Recommend MODAL for the first version. Side-pane is more
  ambitious and has layout-management costs.
- Dialog renders its own Three.js scene (separate canvas /
  WebGL context). Reuses existing sculpture geometry helpers
  from `viz3d_*`.
- ESC closes; click outside closes; "x" in corner closes.
- Open / close are smooth (fade + slight scale).

## Saga step plan (proposed)

1. **affordance**: hover-triggered "?" overlay on 3D sculptures.
   Tracks selected sculpture state. No dialog yet -- just the
   affordance + click handler that for now opens an alert("TODO").
2. **dialog-shell**: modal Yew component (open / close, ESC
   handling, focus trap). Empty body.
3. **single-tensor-view**: render the selected tensor in the
   dialog with its own Three.js scene + the existing sculpture
   builders. Stats panel + axis labels + histogram.
4. **composite-drill**: when the selected sculpture is a chain
   or composite, show the layer breakdown view. Breadcrumb +
   navigation. Pluggable for future composite kinds.
5. **dr-projection**: for hidden-layer drill, add a DR picker
   (UMAP / t-SNE / PCA) routing through existing builtins. The
   dialog re-renders as a 2D scatter (Plotly3D already has a 2D
   mode that can be reused).
6. **encoder-decoder-demo**: a polish step. Add an explicit
   encoder/decoder demo + a tour step pointing at the
   bottleneck affordance.

Estimated 6 steps. Step 5 is the hardest (UMAP latency in the
browser; may need a "compute..." spinner).

## Dependencies on other in-flight work

- The existing `viz3d_*` family in mlpl-web (3 modules) gives us
  the scene + click pipeline. Reuse.
- DR builtins (UMAP, t-SNE, PCA, MDS) already shipped in saga
  33; the dialog calls them as if from the REPL.
- Plotly3D 2D mode (saga 35) handles the projection scatter.
- No dependence on the god-crate-decomposition sagas -- this is
  pure mlpl-web work.

## Open questions

1. Should the dialog persist its DR selection across
   open/close cycles? (Convenience vs. UX consistency.)
2. For composites, do we descend into nested chains (chain of
   chains) recursively, or only one level?
3. What happens for VERY large tensors (e.g. 64x64 images x 1000
   batch = 4M floats)? Render a sample, render lazily, or refuse
   with a hint?
4. Mobile / touch -- does the modal work on phone-sized
   viewports? (Probably degrade gracefully to a list view.)

## Status

**Proposed, not scheduled.** This doc captures the design before
any code lands. Convert to a saga via `agentrail init` when
prioritized.
