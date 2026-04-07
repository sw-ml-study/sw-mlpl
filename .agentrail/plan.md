# MLPL SVG Visualization Saga

## Quality Requirements (apply to EVERY step)

Every step MUST:
1. Follow TDD: write failing tests FIRST, then implement, then refactor
2. Pass all quality gates before committing:
   - cargo test (ALL tests pass)
   - cargo clippy --all-targets --all-features -- -D warnings (ZERO warnings)
   - cargo fmt --all (formatted)
   - markdown-checker -f "**/*.md" (if docs changed)
   - sw-checklist (project standards)
3. Update relevant docs if behavior changed
4. Use /mw-cp for checkpoint process
5. Push immediately after commit
6. Steps that touch the web UI must rebuild pages/ via scripts/build-pages.sh

## Goal

Add an `svg(data, type)` built-in function to MLPL so users can
visualize arrays as inline SVG diagrams in the browser REPL.
This unlocks ~10 ML demos from the learning-series blog (decision
boundary, loss curve, k-means clusters, PCA scatter, attention
heatmap, catastrophic forgetting demo, etc.).

The browser REPL detects an SVG result and renders it inline
instead of as text.

Deployed at: https://sw-ml-study.github.io/sw-mlpl/

## What already exists

- mlpl-array: DenseArray (rank 0/1/2/3+), reductions, matmul, transpose
- mlpl-runtime: 23 built-in functions
- mlpl-eval: AST evaluator
- mlpl-parser: full v1 syntax (literals, arrays, calls, repeat)
- mlpl-repl + mlpl-web: working REPLs
- Display formatting: arrays render as text grids

## Design

### svg() built-in signature

```
svg(data, type_name)            # 2-arg: data + diagram type string
svg(data, type_name, opts)      # 3-arg variant: optional config
```

`type_name` is a string (we'll need string literals or use a
sentinel approach since MLPL has no strings yet -- see decision
below).

### Diagram types (priority order)

1. **scatter** -- 2D points from an Nx2 matrix. Each row is (x, y).
2. **line** -- a vector becomes a polyline; an Nx2 matrix becomes
   (x, y) points connected by lines.
3. **bar** -- a vector becomes a bar chart with one bar per element.
4. **heatmap** -- an MxN matrix becomes a colored grid.
5. **decision_boundary** -- helper that takes a precomputed grid
   of classifier outputs (2D matrix) and renders as a heatmap with
   contour overlay.

These five cover most of the demos in the blog series.

### Output format

`svg()` returns a `DenseArray` carrying the SVG string... but
MLPL has no string type. **Decision needed.**

### CRITICAL DECISION: strings in MLPL

MLPL is array-only. There are no string values. To make svg()
work we need ONE of:

A. Add a String value type to DenseArray (invasive)
B. Add a separate Value enum at the eval layer (Number | String | Array)
   that wraps DenseArray (large refactor across crates)
C. Store strings in a side-channel: svg() returns a special "tagged"
   array (e.g., a 1-d array whose data is a slot id), and the REPL
   looks up the actual SVG string from a session-scoped registry.
   The MLPL eval layer never sees the string. (least invasive)
D. Add a new top-level Value type only in mlpl-eval, leaving
   mlpl-array unchanged. svg() returns a Value::Svg(String), and
   the evaluator's return type becomes Value instead of DenseArray.

**Recommendation: D.** It is the cleanest long-term path because
strings will be needed for LLM client work later (v0.4). It's
mid-sized (touches eval, runtime dispatch, REPL display) but
contained and reversible.

For type_name parameter: introduce string literals to the parser
("scatter") at the same time. Single-quoted or double-quoted.

This means the saga has a foundational first step: introduce a
Value type and string literals. Without this, svg() can't have a
type-name argument.

## Phases

### Phase 1: Foundation (strings + Value type)
- Add Value type to mlpl-eval
- Parser: string literals
- All existing tests still pass (Value::Array(DenseArray) wraps
  the existing return path)

### Phase 2: SVG built-in
- svg() function with scatter type
- Add line, bar, heatmap, decision_boundary types

### Phase 3: REPL integration
- CLI REPL prints raw SVG (or saves to file)
- Browser REPL detects SVG strings and renders inline

### Phase 4: Demos
- Update logistic regression demo with loss-curve viz
- Add k-means demo with cluster viz
- Add attention heatmap demo
- Update tutorial lesson(s) to introduce viz

## Success criteria

- svg([...], "scatter") returns a string starting with <svg
- Browser REPL displays scatter, line, bar, heatmap inline
- Logistic regression demo shows a loss curve below the accuracy
- All tests pass, all quality gates green
- Tutorial has at least one viz lesson
