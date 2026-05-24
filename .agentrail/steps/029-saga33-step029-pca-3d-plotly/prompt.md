Add an interactive Plotly-based 3D viz to the MLPL web playground, then use it for a new PCA-3D demo. Click events on points (and cluster-grouped points) surface the clicked sample's index/label back into the REPL output. Approach A from the chat: HTML/JS payload returned by a new viz type. Approach B (SVG-native interactive 3D) is explicitly rejected.

SCOPE (do everything in this list; defer anything not in it):

1. New viz type `"plotly3d"` in `crates/mlpl-viz`. Signature mirrors the existing `scatter3d`: takes an `[N, 3]` data array. Optional second-arg `aux` for labels (length-N integer or string list, used to color points and group clicks by cluster). Returns a `Value::Str` containing a self-contained HTML fragment: a `<div id="...">` plus a `<script>` block that calls `Plotly.newPlot(...)`. The fragment must be safe to inject into the existing render container -- no `<html>` / `<head>` wrappers.

2. Add Plotly JS as a <script> tag in `apps/mlpl-web/index.html`. Prefer the official CDN (`https://cdn.plot.ly/plotly-2.X.X.min.js`); vendor only if the user explicitly says "no CDN". Add an integrity hash so a CDN swap can't silently change the bundle.

3. Web playground rendering (`apps/mlpl-web/src/handlers.rs` + wherever viz output gets rendered): when an eval result starts with the Plotly fragment's marker comment (suggested: `<!-- mlpl-plotly3d -->`), render the HTML payload as raw innerHTML into a sandboxed container (NOT an iframe -- iframes break Plotly's font and event handling). After insertion, the embedded `<script>` calls `Plotly.newPlot(divId, traces, layout)` and registers a `plotly_click` handler. The click handler appends a one-line "click: sample #N (label=L)" entry to the REPL history via a small JS-to-Rust callback.

4. JS-to-Rust click callback: expose a wasm-bindgen function `mlpl_web::push_plotly_click(sample_index: u32, label: String)` that the script can call. It appends to the demo's history (or to a dedicated "click log" panel if a separate panel is cleaner). Implementer's call which goes -- whichever stays under the existing 25-LOC function budget.

5. Cluster click support: Plotly's `plotly_click` event surfaces the trace index (one trace per cluster label) and the point's within-trace index. The callback maps `(trace_index, point_index)` back to the original sample index using a JS-side `cluster_to_samples[trace][point]` map embedded in the HTML payload.

6. CLI graceful degrade: `mlpl-repl -f script.mlpl` cannot render interactive HTML. Write the HTML payload to `$MLPL_VIZ_CACHE/<hash>.html` (same dir the existing `svg:` payload goes to) and print `viz: <path>` so the user can `open` it in a browser. Document this in `--help` or in `docs/usage.md`.

7. New web demo `PCA_3D` in a new file `apps/mlpl-web/src/demos_dim_reduction.rs`. Wires up: `D = blobs(11, 30, [[0,0,0],[4,4,4],[-4,4,-4]])` (or similar 3D blobs); `X = matmul(D, ...)` to extract coords; `tl = ...` for labels; `proj3 = pca(X, 3)`; `svg(proj3, "plotly3d", tl)`. Intro/takeaway prose mirrors the existing PCA demo's tone -- "rotate, zoom, click any cluster to see which samples land where" -- but DOES NOT duplicate the linear-algebra explanation (link back to PCA demo for the math). KEEP the existing 2D PCA demo untouched; this is a SIBLING demo, not a replacement.

8. Register the new demo file: add `mod demos_dim_reduction;` in lib.rs, add `crate::demos_dim_reduction::PCA_3D` to the DEMOS slice in `apps/mlpl-web/src/demos.rs`.

OUT OF SCOPE (file separately if needed):
- Plotly types other than 3D scatter (no `plotly2d`, no Plotly heatmap, no Plotly surface).
- UMAP, MDS, Random Projection (those are the dim-reduction milestone -- file separately).
- The critical-dimensions heatmap (also the dim-reduction milestone).
- Plotly themes / dark-mode parity with the existing Catppuccin SVG palette (use Plotly defaults; theming is a follow-up).
- Server-side rendering of the HTML to PNG/SVG for the visual regression test suite (deferred; the existing reg-rs harness from saga 33 step 026 cannot rasterize Plotly JS-driven output -- the regression test for this demo can be a simpler "HTML fragment contains the expected sample count + trace count" assertion).

QUALITY GATES (saga 33 standard):
1. `cargo test --release` -- new viz type emits parseable HTML; sample-count and trace-count assertions in `mlpl-viz/tests/`.
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
3. `cargo fmt --all -- --check`.
4. `markdown-checker` for any docs added (the `docs/usage.md` note + the new demo's intro/takeaway, if those use any nontrivial markdown).
5. `sw-checklist` net-negative on FAILs and warnings.
6. Run `scripts/build-pages.sh` to rebuild `pages/` so the github.io live demo picks up the new viz + demo. Commit `pages/` in the same commit.
7. Push.

VERIFICATION:
- `./scripts/serve.sh` (release default after step 027). Open http://localhost:9957/. Click "PCA 3D" demo. Confirm:
  (a) Plotly 3D scatter renders with 3 colored clusters (or however many `blobs` were spawned).
  (b) Mouse drag rotates the view; scroll zooms; double-click resets.
  (c) Single-click on a point appends a "click: sample #N (label=L)" entry into the REPL history.
  (d) Click on a different cluster surfaces a different label.
  (e) The existing 2D PCA demo still works exactly as before.

DELIVERABLE: one saga step that lands an interactive 3D viz primitive plus the first demo to use it. Future steps can add more `plotly3d` consumers without re-doing the JS-to-Rust glue.
