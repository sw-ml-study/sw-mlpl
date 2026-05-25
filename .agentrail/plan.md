# 3D Visualization Stage Milestone

Saga 35, proposed.

## Vision

A parallel 3D viewport runs alongside the REPL, showing
intermediate computation values as sculptural objects on a
persistent "stage." The stage is a large horizontal
landscape (salt flat, distant mountains) where each demo
step occupies a spatial position -- a 3D storyboard of the
computation's history.

The camera pans along the timeline to show past steps
(frozen sculptures) and the current step (possibly
animating). Individual values -- scalars, vectors, matrices,
tensors -- are rendered as 3D forms whose shape, rank,
size, and structure are visually apparent.

Think of a game's avatar sprite strip: the full stage is a
large pre-composed 3D scene, and the viewport shows a
window into the section relevant to the current step.

## Non-goals (initial pass)

- **Accurate tensor visualization.** The first pass uses
  labeled placeholder boxes (shape + rank text labels on
  colored cubes/planes). Accurate heatmap textures,
  element-level detail, and animated transforms come in
  later iterations.
- **Every demo covered.** Initial integration covers 1-2
  demos (e.g., Basics and a simple training loop) as proof
  of concept. Full coverage is iterative.
- **Mobile support.** WebGL/Three.js on mobile is
  unreliable. The 3D view is desktop-only; mobile hides
  the toggle.
- **Persistent 3D assets.** The landscape/stage is
  procedurally generated or loaded from a simple GLTF
  asset, not a large pre-baked world.

## Architecture

### Technology

**Three.js** via JavaScript interop (wasm-bindgen) for the
3D scene: scene graph, PerspectiveCamera, OrbitControls,
lighting, geometry, GLTF loading.

**D3.js** for data-driven animation of many moving data
points within each sculpture. When a weight matrix updates
during training, D3 transitions drive per-element color
and height changes across hundreds of cells simultaneously.
Three.js owns the 3D camera and stage layout; D3 owns the
per-element animation within each sculpture's texture or
instanced mesh. D3 is also the right tool for 2D overlays
(axis labels, value histograms) rendered as HTML layers
positioned relative to 3D objects via CSS2DRenderer.

### Integration pattern

```
+------------------+     +-------------------+
|  Yew App (WASM)  |     | Three.js (JS)     |
|                  |     |                   |
|  REPL eval       |---->| scene.add(mesh)   |
|  demo runner     |     | camera.pan(step)  |
|  step events     |     | animate()         |
|                  |<----| click events      |
+------------------+     +-------------------+
        ^                        ^
        |                        |
   Yew state handles     <canvas> element
   (show_3d, step_idx)   managed by Three.js
```

**Communication:** Yew owns the eval pipeline and emits
"step events" (step index, value shapes, operation name)
to JS via `wasm_bindgen` exports. The JS side manages the
Three.js scene, adds/updates meshes, and moves the camera.
Click events on 3D objects emit back to Yew via
`CustomEvent`.

**The JS module** lives in `apps/mlpl-web/js/stage3d.js`
(or a small npm-free bundle). Loaded via a `<script>` tag
with `type="module"`. Three.js loaded from CDN or vendored.

### Toggle mechanism

- **REPL command:** `:3d` toggles the 3D viewport on/off.
  `:3d on` / `:3d off` for explicit control.
- **Hotkey:** `Ctrl+3` toggles (mirrors the REPL command).
- **State:** `show_3d: UseStateHandle<bool>` in UiState,
  default false.
- **Layout:** When on, the main area splits into a 2-pane
  layout (REPL left, 3D viewport right) or the 3D view
  replaces the output area. The split mirrors the existing
  tutorial split pattern.

### Step event protocol

Each eval step emits a `Stage3dEvent` to the JS side:

```rust
pub struct Stage3dEvent {
    pub step_idx: usize,
    pub label: String,        // "matmul(W, x)"
    pub inputs: Vec<ShapeInfo>,
    pub output: ShapeInfo,
}

pub struct ShapeInfo {
    pub name: String,    // "W"
    pub shape: Vec<usize>, // [3, 5]
    pub rank: usize,
    pub element_count: usize,
}
```

The JS side maps each event to a 3D "sculpture" placed at
`x = step_idx * spacing` on the stage. Inputs are rendered
as smaller objects flowing into the operation; the output
is rendered as the result object.

### Stage landscape

The "salt flat" ground plane extends along the x-axis.
Mountains or a skybox provide depth cues. Each step gets
a position along x. The camera starts at step 0 and
tracks the current step. OrbitControls allow the user to
freely look around; double-click snaps back to the current
step.

Initial implementation: a white PlaneGeometry ground with
grid lines, no mountains. The landscape can evolve
independently of the step visualization logic.

## Phases

### Phase 1: Infrastructure (toggle + canvas + Three.js bootstrap)

**Deliverables:**

1. `show_3d` state handle in UiState, toggled by `:3d`
   REPL command and Ctrl+3 hotkey.
2. When `show_3d` is true, render a `<canvas id="stage3d">`
   element in a split pane alongside the REPL output.
3. `apps/mlpl-web/js/stage3d.js`: Three.js bootstrap
   (scene, camera, renderer, OrbitControls, ground plane,
   ambient light). Renders an empty stage.
4. Yew calls `window.__stage3d_init(canvas)` on mount and
   `window.__stage3d_destroy()` on unmount.
5. Trunk `<link data-trunk rel="copy-file">` for the JS
   module + Three.js vendor bundle.

**No step events yet.** Just the empty stage with camera
controls.

### Phase 2: Step event pipeline

**Deliverables:**

1. `Stage3dEvent` + `ShapeInfo` structs in a new
   `viz3d_events.rs` module.
2. Eval pipeline emits events via
   `window.__stage3d_add_step(json)` after each REPL
   eval or demo line.
3. JS side receives events and places a labeled
   placeholder box at `(step_idx * 2, 0, 0)`.
4. Camera auto-pans to the latest step.

**Placeholder rendering:** A colored `BoxGeometry` with
`TextSprite` label showing `"matmul [3,5] -> [3,3]"`.
Color encodes the operation type (blue = matmul, green =
activation, red = loss, gray = assignment).

### Phase 3: Shape-aware sculptures

Replace placeholder boxes with shape-proportional meshes:

- Scalar: small sphere
- Vector `[N]`: horizontal bar, length proportional to N
- Matrix `[M, N]`: flat rectangle, M x N proportional
- Rank-3+ tensor `[B, M, N]`: stacked rectangles (B
  layers)

Labels show `name: [shape]` and `rank R, N elements`.

### Phase 4: Demo integration

Wire up 2-3 demos to emit step events:

1. **Basics** -- simple arithmetic, arrays, reshape.
   Shows scalars flowing into vectors flowing into
   matrices.
2. **Loss Curve** -- training loop iterations. Each
   iteration is a step; loss value shown as a vertical
   bar whose height tracks the loss.
3. **Moons MLP** -- forward pass through layers. Each
   layer is a transform sculpture (input -> weights ->
   output).

### Phase 5: Animation + transitions

Current-step sculpture animates:
- Matrix multiply: input matrices slide together, output
  grows from their intersection.
- Activation function: elements visually shift (e.g.,
  negative elements squash to zero for ReLU).
- Reshape: elements rearrange into the new shape.

Past steps freeze in place as the camera advances.

### Phase 6: Landscape + polish

- GLTF landscape asset (salt flat + distant mountains)
  or procedural skybox.
- Fog for depth.
- Shadow casting from sculptures onto the ground.
- Click a past sculpture to inspect its value in the
  REPL (emit `:describe <var>` via CustomEvent).

## Quality requirements

Same as saga 34. TDD where possible (event structs and
shape-to-mesh mapping are pure and testable). JS code
tested via manual verification in the browser. Each phase
is independently deployable with the feature toggled off
by default.

The Three.js dependency is loaded only when the 3D view
is activated (lazy-load the module on first `:3d on`).
No bundle-size impact when the feature is off.

## Module budget plan

| New module | Concern | Est. fns | Est. LOC |
|---|---|---|---|
| `viz3d_toggle.rs` | `:3d` command + Ctrl+3 handler | 2 | ~30 |
| `viz3d_events.rs` | Stage3dEvent + ShapeInfo structs | 2 | ~40 |
| `viz3d_panel.rs` | Canvas element + split layout | 2 | ~40 |
| `js/stage3d.js` | Three.js scene management | JS | ~200 |

All Rust modules under 4-fn warning target.
