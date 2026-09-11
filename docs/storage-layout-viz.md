# Storage-layout 3D visualization -- sw-mlpl's part

Status: DESIGN / coordination doc for a three-repo demo (planning doc; it
may name phases and cross-repo plans). Source vision:
`../demo-extensions/docs/research.txt`.

## The demo, in one line

An interactive 3D view of a system's storage/memory layout drawn as stacks
of classified blocks (the FlashViz metaphor). The first producer is the
SWTOS two-stage storage/runtime architecture (stored image -> provider ->
validated C24IMG -> allocated process -> reclaimed memory). Additional
producers of the SAME contract: MLOS (`../../sw-ml-study/sw-os-ml`) and
MesaOS (`../../softwarewrighter/MesaOS`). The visualizer is deliberately
system-agnostic -- any OS that emits the columnar layout below can be shown
through identical MLPL code.

## Architecture and the repos

```
sw-tos / sw-os-ml (MLOS)   storage semantics   (each OS's own agent)
  | emit
  v
storage-layout.json        the data contract   (COLUMNAR; see below)
  |
  v
MLPL viz script            visualization        (sw-mlpl -- THIS repo)
  | array-programming: layout math, classify, assemble geometry
  v
native3d extension         graphics             (../demo-extensions agent)
  |
  v
3D viewer
```

Boundary (kept deliberately narrow so each piece stays reusable):

- The OS (SWTOS or MLOS) knows storage semantics -- it emits the layout,
  nothing about pixels.
- MLPL knows visualization semantics -- it turns the layout into geometry.
- native3d knows graphics -- it draws bulk boxes/labels and does picking.

sw-mlpl's part is the array-programming MIDDLE. It does NOT understand any
particular OS and it does NOT draw; it consumes a structured layout and
produces geometry arrays. Nothing in the contract or the MLPL viz code is
SWTOS-specific -- region kinds/owners are data from the JSON, and the color
palette (a visualization concern MLPL owns) covers the union of the
vocabularies the producers emit (or is chosen per producer).

## The data contract: COLUMNAR (struct-of-arrays)

Decision (maintainer, 2026-09-11): the layout is columnar -- each field is a
homogeneous array or list of strings. This is idiomatic for an array
language and, crucially, `parse_json` already ingests it today (homogeneous
numeric arrays -> a numeric array value; all-string arrays -> a string
list). A row-oriented array-of-objects would instead require a new
list-of-records language feature; that is deliberately out of scope here and
tracked separately.

Every `region_*` array has the same length N (one entry per region). Regions
are grouped into named spaces (flash, EBR/RAM, system RAM); a `region_space`
column joins each region to the `spaces` metadata by name.

```json
{
  "schema": "sw-ml-study.system-layout",
  "version": 1,
  "provenance": { "producer": "sw-tos", "revision": "c48a479" },

  "spaces":          ["flash", "ebr", "sysram"],
  "space_name":      ["W25Q32 storage", "runtime EBR", "system RAM"],
  "space_block":     [8, 8, 8],
  "space_capacity":  [4194304, 262144, 65536],

  "region_id":       [1, 2, 3, 4, 5],
  "region_space":    ["flash", "flash", "flash", "ebr", "ebr"],
  "region_kind":     ["header", "catalog", "image", "text", "stack"],
  "region_name":     ["storage header", "catalog records",
                      "embedded-hello", "hello text", "hello stack"],
  "region_owner":    ["kernel", "kernel", "hello", "hello", "hello"],
  "region_start":    [0, 8, 128, 0, 4096],
  "region_length":   [8, 120, 44, 512, 1024],

  "region_text_words": [0, 0, 6, 0, 0],
  "region_data_words": [0, 0, 0, 0, 0],
  "region_bss_words":  [0, 0, 0, 0, 0],

  "rel_kind":        ["describes", "loads-to"],
  "rel_from":        [2, 3],
  "rel_to":          [3, 5]
}
```

Rules:

- `schema` is the constant `"sw-ml-study.system-layout"`; `version` is the
  contract version; `provenance` is a small object carrying at least the
  `producer` and its source `revision` (so a rendered snapshot is traceable).
  These are the identifying header the consumers requested.
- All `region_*` arrays are length N and index-aligned. `region_id` is an
  explicit, STABLE integer id per region (assigned by the producer, stable
  across snapshots) -- it is what native3d uses for picking and
  cross-highlighting, so it must not be a positional afterthought.
- All `space_*` arrays are length S and index-aligned; `spaces[i]` is the key.
- `region_space[j]` is one of `spaces`. `region_kind` / `region_owner` are
  from small closed vocabularies (see color modes below).
- Offsets and lengths are in bytes. `space_block` is the block size in bytes
  (8 for SWTOS). Image word-counts are 24-bit words (0 for non-image kinds).
- Relationships are an EDGE table of length E (independent of N):
  `rel_kind` (e.g. `describes`, `loads-to`), `rel_from` and `rel_to`
  (`region_id` values). They may be empty initially; the "explain selected
  program" view (catalog -> extent -> C24IMG -> allocation) consumes them.
- Emit from the SAME build artifacts the producer already uses (avoid drift).

A worked sample lives at `examples/viz/storage-layout.json`.

## What sw-mlpl does with it (the reference script)

Pure array-programming, no loops over pixels:

1. Parse: `layout = parse_json(read_text("storage-layout.json"))`, then read
   the columns with `record_get`.
2. Block geometry: map each region's byte extent to logical cells (1 cell =
   `space_block` bytes) and lay cells out in 16x16 layers:
   `x = block % 16`, `z = (block / 16) % 16`, `y = block / 256`. `mod`,
   `floor` and `/` are elementwise, so this runs over the whole block-index
   vector at once. Assemble `centers [N,3]` and `sizes [N,3]` with `reshape`
   / `concat`.
3. Classify: map `region_kind` (a string list) to an RGBA row per region ->
   `colors [N,4]`; join `region_space` to `space_block`/`space_capacity` for
   per-space placement. This string-list -> row lookup is the one capability
   gap (see below).
4. Identify: a stable `ids [N]` vector for picking / cross-highlighting.
5. Drive: push `centers`, `sizes`, `colors`, `ids` to native3d via the
   extension surface (all shipped: rank-2 arrays marshal both directions).

Color means one thing at a time (modes, per the research): Purpose
(header/catalog/text/data/bss/stack/state/free), Owner (kernel/shell/child),
Location (RAM/EBR/flash/SD), State (free/stored/loading/live/reclaimable).
Selection adds an outline, not a color change.

## Capability status (from the sw-mlpl scope)

SHIPPED and sufficient:

- `parse_json` for the columnar shape (homogeneous arrays / string lists).
- Nested field access: `record_get`, `has_field`, `record_keys`.
- String-list indexing: `list_len`, `list_get`.
- Layout math: `mod`, `floor`, `ceil`, `/` (integer div = `floor(a / b)`),
  all elementwise.
- Geometry assembly: value-level `reshape` and axis-wise `concat` build
  `[N,3]` / `[N,4]`.
- Extension boundary: `load_extension`, `MLPL_EXTENSION_PATH`, namespaced
  invoke, and C-ABI array marshaling for rank 1..8 in BOTH directions --
  `[N,3]` centers and `[N,4]` colors cross to native3d and results come back.

The one gap (now RESOLVED):

- Vectorized string classification -- mapping a `region_kind` string list to
  RGBA rows -- had no idiomatic primitive (`each`/`table` are numeric-only,
  and `take` cannot row-select). Shipped as `select_rows(table, keys)`: a
  record of `key -> row` plus a string list of keys yields the `[N,C]` matrix
  of looked-up rows. `colors = select_rows(palette, region_kind)`.

## native3d invocation surface

native3d is a companion NATIVE extension (wgpu/winit) provided by
`../demo-extensions`; it is not built in this repo. sw-mlpl provides only the
host load/registry/marshal surface. The research proposes reusable
primitives (not SWTOS-specific): `set_boxes(centers, sizes, colors, ids)`,
`set_labels(positions, strings, ids)`, `pick(x, y) -> id`, `camera(...)`,
`set_visibility(ids, ...)`, `set_highlight(ids, ...)`. sw-mlpl's reference
script targets exactly these.

A self-contained fallback that sw-mlpl CAN run without the native extension:
a 2D orthographic memory-map rendered with the shipped SVG renderer, from the
same columnar model -- useful as a web-playground demo while native3d lands.

## Phasing (mirrors the research)

- Phase 1 -- static generated layout: storage header, catalog records,
  embedded image extents, padding/free. Proves the pipeline.
- Phase 2 -- runtime snapshot: EBR allocation, executable boundary,
  high-water mark, process state, stack, free.
- Phase 3 -- TUI-linked: te-rs streams snapshot events
  (process-spawn/image-load/allocation/exit/reclaim); the viewer animates.

## Who provides what

- `../../sw-embed/sw-tos`: `storage-layout.py` (and later a runtime emitter)
  producing `storage-layout.json` in THIS columnar contract, from the build
  artifacts.
- `../../sw-ml-study/sw-os-ml` (MLOS) and `../../softwarewrighter/MesaOS`
  (MesaOS): additional producers of the SAME columnar contract for their own
  storage/memory layouts, when their agents emit that data. The visualizer
  treats them identically; only the region-kind/owner vocabulary (and hence
  the palette coverage) may differ.
- sw-mlpl (this repo): the reference viz library (`examples/viz/`), the one
  classification primitive, and the 2D orthographic SVG fallback demo. Kept
  system-agnostic so both SWTOS and MLOS layouts render through the same code.
- `../demo-extensions`: native3d filled boxes + picking + labels + camera.

## Handoff status (2026-09-11)

sw-mlpl's part of the pipeline is built, tested, and on `main`:

- Contract locked (columnar; `schema` / `provenance` / `region_id` /
  relationship edge table) -- this section and the JSON above are canonical.
- `select_rows(table, keys)` builtin shipped (the classification primitive).
- Geometry library `examples/viz/layout.mlpl` (block layout math, per-region
  boxes, palette classification, stable ids) with tests. `u:default_palette()`
  covers the closed region-kind vocabulary (header/catalog/image/free/padding/
  text/data/bss/state/stack/kernel); `select_rows` is strict, so a new kind
  needs a palette row (a test pins the vocabulary).
- Two runnable surfaces: `examples/viz/memory_map_2d.mlpl` (self-contained 2D
  heatmap, no native extension) and `examples/viz/render_native3d.mlpl` (the
  3D reference driving native3d).
- Conforming sample artifact `examples/viz/storage-layout.json`, sha256
  `b5a328a623fd7b8378bf47b7014318525334495a9911354539a007f0fbfa657b` (a
  hand-written sample; SWTOS now emits the real artifact, so build against
  that when present).

Producer status:

- `sw-tos`: SHIPPED the emitter (sw-tos commit `5c4a24e`) and a real artifact
  (sha256 `58d29496c75efaa95567d0208ead48dc4c83797c86f2a6952dfc78d1ce748a84`,
  artifact commit `a08d886`) carrying all four contract additions. Its real
  data exercises the `padding` kind, which is now in `u:default_palette()`.
- `sw-os-ml` (MLOS), `MesaOS`: emit `storage-layout.json` in the locked
  columnar contract from their own build artifacts; report commit + checksum.
- `demo-extensions`: implement the generic native3d primitives
  (`set_boxes` / `set_labels` / `pick` / `camera` / `set_visibility` /
  `set_highlight`); the reference script targets exactly these.
