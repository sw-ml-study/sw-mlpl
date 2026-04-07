# Visualization Milestone (v0.3.0)

Building on the ML foundations milestone, v0.3 adds inline SVG
visualization to MLPL. Arrays can now be rendered as scatter plots,
line charts, bar charts, heatmaps, and decision-boundary surfaces,
and the browser REPL displays the SVG inline beneath the input.

## Delivered

- [x] String literals (`"scatter"`, `"line"`, ...)
- [x] `Value` type in `mlpl-eval` (Array | Str) wrapping the existing
      `DenseArray` return path
- [x] `mlpl-viz` crate with `render(data, type)` and
      `render_with_aux(data, type, aux)` dispatch
- [x] `svg(data, type)` and `svg(data, type, aux)` built-in
- [x] Diagram types: `scatter`, `line`, `bar`, `heatmap`,
      `decision_boundary`
- [x] `grid(bounds, n)` runtime built-in for 2D evaluation grids
- [x] Browser REPL renders SVG output inline via
      `Html::from_html_unchecked`
- [x] CLI REPL prints `[svg: N bytes]` summary, with `--svg-out <dir>`
      to save each SVG to a file
- [x] New demos: `loss_curve.mlpl`, `decision_boundary.mlpl`
- [x] `logistic_regression.mlpl` updated with `svg()` calls
- [x] Web demo dropdown alphabetized; "Decision Boundary",
      "Loss Curve", and "Visualizations" entries added
- [x] Tutorial Lesson 11 "Visualizing Data" with one example per
      diagram type
- [x] `docs/lang-reference.md`, `docs/usage.md`, and `README.md`
      updated for the new features

## Demos

```bash
cargo run -p mlpl-repl -- -f demos/loss_curve.mlpl
cargo run -p mlpl-repl -- -f demos/decision_boundary.mlpl
cargo run -p mlpl-repl -- -f demos/logistic_regression.mlpl
```

Or try them in the [browser REPL](https://sw-ml-study.github.io/sw-mlpl/).

## What's Next

- [ ] Higher-level analysis helpers (saga step 009) that emit
      multi-panel SVG summaries from a single call
- [ ] Random array constructors for synthetic datasets
- [ ] PCA / k-means / attention demos using the new viz primitives
