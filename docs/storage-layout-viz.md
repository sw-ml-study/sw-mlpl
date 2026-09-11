# Storage-layout 3D visualization -- sw-mlpl's part

Status: DESIGN / coordination doc for a three-repo demo (planning doc; it
may name phases and cross-repo plans). Source vision:
`../demo-extensions/docs/research.txt`.

## The demo, in one line

An interactive 3D view of a system's storage/memory layout drawn as stacks
of classified blocks (the FlashViz metaphor). The first producer is the
SWTOS two-stage storage/runtime architecture (stored image -> provider ->
validated C24IMG -> allocated process -> reclaimed memory); MLOS
(`../../sw-ml-study/sw-os-ml`) is a second producer of the SAME contract, so
the visualizer is deliberately system-agnostic -- any OS that emits the
columnar layout below can be shown.

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
  "version": 1,
  "spaces":          ["flash", "ebr", "sysram"],
  "space_name":      ["W25Q32 storage", "runtime EBR", "system RAM"],
  "space_block":     [8, 8, 8],
  "space_capacity":  [4194304, 262144, 65536],

  "region_space":    ["flash", "flash", "flash", "ebr", "ebr"],
  "region_kind":     ["header", "catalog", "image", "text", "stack"],
  "region_name":     ["storage header", "catalog records",
                      "embedded-hello", "hello text", "hello stack"],
  "region_owner":    ["kernel", "kernel", "hello", "hello", "hello"],
  "region_start":    [0, 8, 128, 0, 4096],
  "region_length":   [8, 120, 44, 512, 1024],

  "region_text_words": [0, 0, 6, 0, 0],
  "region_data_words": [0, 0, 0, 0, 0],
  "region_bss_words":  [0, 0, 0, 0, 0]
}
```

Rules:

- All `region_*` arrays are length N and index-aligned.
- All `space_*` arrays are length S and index-aligned; `spaces[i]` is the key.
- `region_space[j]` is one of `spaces`. `region_kind` / `region_owner` are
  from small closed vocabularies (see color modes below).
- Offsets and lengths are in bytes. `space_block` is the block size in bytes
  (8 for SWTOS). Image word-counts are 24-bit words (0 for non-image kinds).
- Emit from the SAME build artifacts SWTOS already uses (avoid drift).

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

The one GAP:

- Vectorized string classification: mapping a `region_kind` string list to
  RGBA rows (and joining `region_space` to space metadata) has no idiomatic
  primitive today -- `each`/`table` are numeric-only. It is expressible with
  a manual `while` + `list_get` + `str_eq` loop, but that is verbose. The
  `classify-primitive` step adds the smallest general primitive (a
  string-list -> category-index map and/or gather-rows-by-index), not a
  demo-specific builtin.

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
- `../../sw-ml-study/sw-os-ml` (MLOS): a second producer of the SAME columnar
  contract for its own storage/memory layout, when its agent emits that data.
  The visualizer treats it identically; only the region-kind/owner vocabulary
  (and hence the palette coverage) may differ.
- sw-mlpl (this repo): the reference viz library (`examples/viz/`), the one
  classification primitive, and the 2D orthographic SVG fallback demo. Kept
  system-agnostic so both SWTOS and MLOS layouts render through the same code.
- `../demo-extensions`: native3d filled boxes + picking + labels + camera.
