# Saga: storage-layout-viz

sw-mlpl's part of a three-repo coordinated demo: an interactive 3D view of a
system's storage/memory layout as stacked blocks (see
`../demo-extensions/docs/research.txt`).

Architecture: `sw-tos -> storage-layout.json -> MLPL viz script -> native3d
extension -> 3D viewer`. Boundary: SWTOS knows storage semantics; MLPL knows
visualization semantics; native3d knows graphics.

sw-mlpl provides the ARRAY-PROGRAMMING middle. The data contract is COLUMNAR
(struct-of-arrays; maintainer decision 2026-09-11): each field is a homogeneous
array or StrList, which `parse_json` already handles. MLPL consumes it, does the
block->(x,y,z) layout math, classifies regions to RGBA, assembles geometry
([N,3] centers/sizes, [N,4] colors, [N] ids), and drives native3d via the
shipped extension array-marshaling surface (rank 1..8, both directions).

Parallel repos: `../demo-extensions` extends native3d (filled boxes + picking);
`../../sw-embed/sw-tos` emits the columnar JSON.

## Steps

1. contract-and-design -- docs/storage-layout-viz.md: the columnar JSON
   contract sw-tos emits (fields, types, block_size, capacity), sw-mlpl's role
   and boundary, the reference-script architecture (layout math + classify +
   assemble + drive), the capability gap analysis (parse_json handles the
   columnar shape; the one gap is vectorized string->RGBA classification), and
   the native3d invocation surface. The coordination artifact the sw-tos and
   demo-extensions agents key off. Ship a small sample columnar
   storage-layout.json fixture under examples/viz/.

2. classify-primitive -- the one language gap: mapping a StrList of region
   kinds to RGBA rows. Add the SMALLEST general primitive that makes it
   idiomatic (e.g. gather-rows-by-index and/or StrList->category-index), not a
   demo-specific builtin; only if the existing while-loop form is too verbose.
   TDD, tests/clippy/fmt, sw-checklist anticipated.

3. geometry-lib -- examples/viz/layout.mlpl (+ colors/geometry helpers): pure
   MLPL turning the columnar layout into geometry -- block->[x,y,z] layers
   (x=b%16, z=(b/16)%16, y=b/256), per-region centers/sizes [N,3], kinds ->
   [N,4] RGBA, stable ids [N]. Every def carries a docstring; formatted. TDD
   against the fixture (shapes + spot values via a pinned test).

4. render-and-drive -- (a) a self-contained sw-mlpl-owned runnable surface: a
   2D orthographic memory-map rendered via the shipped SVG renderer (the doc's
   "2D orthographic mode, same semantic model"), suitable for a web-playground
   demo NOW; (b) a native3d reference script calling its box/label primitives
   with the assembled arrays, documented + gated (native3d is a companion
   extension built downstream). Docs.

5. coordinate-and-close -- relay the contract to the sw-tos and demo-extensions
   agents (what each provides/consumes), update the wiki + docs, refresh
   CHANGES.md, mark shipped in docs/future-sagas-queue.md. `--done`.
