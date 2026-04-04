# Viz Contract

## Purpose

Define the behavioral spec for trace visualization in MLPL. `mlpl-viz`
renders trace data into visual formats. This is post-MVP scope -- MVP
uses CLI + JSON trace export only.

## Key Types and Concepts

### Renderer

Takes a `Trace` and produces visual output.

- Timeline view: step-by-step execution replay
- Array view: 2-D heatmap or table for array values
- Future: Yew/WASM interactive viewer

## Invariants

- Rendering must not modify the trace
- Visual output must faithfully represent trace data

## What This Contract Does NOT Cover

- Trace recording (that is `mlpl-trace`)
- Evaluation logic
- CLI output formatting (that is in apps)
- WASM/Yew build infrastructure (post-MVP)

## Open Questions

- Output formats: SVG? HTML? Terminal-based?
- Whether viz is a library or an app
- When the Yew/WASM viewer enters scope
