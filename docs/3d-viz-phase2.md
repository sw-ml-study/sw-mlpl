# 3D Visualization Phase 2: Scale, Structure, and Algorithmic Animation

Phase 1 (saga 35) delivered the stage, shape-aware sculptures,
demo integration, and click-to-inspect. Phase 2 makes the
3D view informative -- showing relative scale, data flow
connections, internal array structure, and animated
algorithmic operations.

## Principles

1. **Show relative scale.** A scalar and a `[768, 768]`
   matrix must coexist visibly. Log-proportional sizing
   with a graduated legend.
2. **Show data flow.** Which step's output became the next
   step's input? Connection arrows.
3. **Show internal structure.** Weight matrices as colored
   grids, attention scores as heatmaps, loss as a height
   bar.
4. **Animate algorithms.** Matmul, softmax, backprop,
   reshape -- each has a characteristic motion.

## Phase 2a: Scale + connections + legend

**No element data needed.** Uses existing shape metadata.

### Log-proportional sizing

Current: sculpture dimensions are linear-proportional to
array dimensions. A `[3, 4]` matrix and a `[300, 400]`
matrix differ by 100x in visual size.

Proposed: sculpture dimensions proportional to
`log2(dim)`. A `[3]` vector has width ~1.6, a `[300]`
vector has width ~8.2 -- visible difference but not
overwhelming.

Minimum size: 0.3 units (scalars). Maximum size: 6 units
(largest dimension capped to keep the stage navigable).

### Graduated legend

A fixed bar on the ground plane (or floating along the
bottom of the viewport) showing:

```
|---1---10---100---1K---10K---100K---|
```

Each mark is a small reference cube at that scale. The
legend auto-scrolls with the camera.

### Connection arrows

When step N's label references a variable from step M
(e.g., step 3 is `y = matmul(W, x)`, step 1 is `x =
iota(6)`, step 2 is `W = randn(42, [3, 6])`), draw a
thin curved line from step 1's sculpture to step 3, and
from step 2 to step 3.

Implementation: parse the label for variable references,
match against previously emitted step names, render as
`THREE.TubeGeometry` following a quadratic bezier curve
(lifted slightly above the ground to avoid z-fighting).

## Phase 2b: Element-level data pipeline

Extend `Stage3dEvent` to carry actual array data:

```rust
pub struct Stage3dEvent {
    pub step_idx: usize,
    pub label: String,
    pub output: ShapeInfo,
    pub values: Option<Vec<f64>>,  // NEW
}
```

For small arrays (<=1000 elements): send all values.
For large arrays: send a statistical summary:

```rust
pub struct ArraySummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub histogram: Vec<usize>,  // 16-bin histogram
    pub sample: Vec<f64>,       // 64 evenly-spaced samples
}
```

The WASM session's eval returns a string; to get actual
values we need either:
- A new `eval_values(input) -> (String, Option<Vec<f64>>)`
  method on WasmSession, or
- Parse the output string back into floats (fragile but
  no API change)

Recommendation: add `eval_with_values` to WasmSession that
returns both the display string and the raw `DenseArray`.

## Phase 2c: Structured visualization by operation type

Each operation type gets a custom 3D representation:

### Scalars
Small sphere. Color encodes value: blue (negative) ->
white (zero) -> red (positive). Size: fixed 0.3 units.

### Vectors
Horizontal bar chart. Each element is a thin vertical
column whose height = value (clamped). Color gradient
from blue (min) to red (max). Bar group width = log2(N).

### Matrices (weight matrices, attention scores)
Flat grid of colored cells. Each cell's color = value
(blue-white-red diverging colormap). Grid dimensions =
log2(M) x log2(N). Hover/click a cell to see its
coordinates and value.

For attention score matrices: opacity encodes magnitude
so the "attention pattern" is immediately visible.

### Loss values (training)
Vertical bar rising from the ground. Height = loss value
(log scale for large losses). Color shifts from red (high
loss) to green (low loss) across training steps. Multiple
training steps form a loss landscape.

### Neural net layers
Input vector (bar) -> weight grid -> output vector (bar).
Arrows connect input elements through weight rows to
output elements. During training animation, the weight
colors shift frame by frame.

### Attention mechanism
Full pipeline visualization:
1. Q, K, V matrices appear as three colored grids
2. Q and K transpose-multiply (animated: Q slides toward
   K^T, product grid grows from their intersection)
3. Softmax applies (colors shift to the softmax
   distribution -- most cells fade, attended cells glow)
4. Weighted V produces the output (arrows from bright
   softmax cells to V rows to output)

## Phase 2d: Animated algorithms

D3.js drives per-element transitions within Three.js
sculptures:

### Matmul animation
Input matrices slide toward each other. The output
grid grows cell by cell, each cell lighting up as its
dot product completes. Speed: 1-2 seconds for the
full animation, skippable.

### Softmax animation
Pre-softmax scores shown as bar heights. Exponentiation
stretches the bars. Normalization compresses them to sum
to 1. Color shifts from raw-score blue to probability
green.

### Backprop animation
Gradient flows backward through the graph. Each
connection arrow pulses from output to input. Weight
matrices show per-element gradient magnitude as
overlay color (transparent -> opaque red for large
gradients).

### Reshape animation
Elements visually rearrange from old shape to new
shape. A `[12]` vector's 12 bars fold into a `[3, 4]`
grid. Each element moves to its new position over
0.5 seconds.

### Training loop animation
Loss bar descends step by step. Weight grids shift
colors between steps. Learning rate shown as the
"speed" of color change (fast early, slow late with
decay).

## Phase 2e: Interaction enhancements

### Hover tooltips
Hover over any sculpture element (matrix cell, vector
bar, connection arrow) to see coordinates, value, and
gradient in a floating label.

### Time scrubber
A horizontal slider below the nav buttons. Drag to
scrub through steps -- sculptures appear/disappear
as the scrubber moves. The 3D equivalent of the
REPL's scrollback.

### Freeze/play
Toggle between:
- **Play:** new steps auto-append and camera follows
- **Freeze:** camera stays put, new steps build off-
  screen (arrow to latest still works)

### Split-view inspection
Click a sculpture to "zoom in" -- it expands to fill
the viewport with full element-level detail while the
rest of the stage dims. Click again or press Escape to
return to the timeline view.

## Quality requirements

Same as phase 1. Each sub-phase is its own saga with
independently deployable steps. Element data pipeline
(2b) is the hard prerequisite for everything after 2a.

## Suggested saga ordering

1. **Saga 36:** Scale + connections + legend (2a)
2. **Saga 37:** Element data pipeline (2b)
3. **Saga 38:** Structured viz by op type (2c)
4. **Saga 39:** Animated algorithms (2d)
5. **Saga 40:** Interaction enhancements (2e)

Each saga is 5-8 steps. Total: ~30 steps across 5 sagas.
The value compounds: each saga makes the previous one's
output more informative.
