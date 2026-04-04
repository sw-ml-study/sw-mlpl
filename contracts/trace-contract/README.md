# Trace Contract

## Purpose

Define the behavioral spec for execution tracing in MLPL. `mlpl-trace`
records each evaluation step so that execution can be replayed,
visualized, and debugged. Traceability is a first-class concern.

## Key Types and Concepts

### TraceEvent

A single recorded evaluation step.

- Source span (what code produced this)
- Operation name or description
- Input values (snapshots or references)
- Output value
- Timestamp or sequence number

### Trace

An ordered collection of TraceEvents for one evaluation.

- Can be serialized to JSON for export
- Can be replayed step-by-step

## Invariants

- Trace events are ordered by evaluation sequence
- Every event references a valid source span
- Trace serialization is deterministic (same eval -> same JSON)
- Tracing must not alter evaluation results

## Error Cases

- `TraceError` is local to `mlpl-trace`
- Serialization failures
- Invalid span references

## What This Contract Does NOT Cover

- Visual rendering (that is `mlpl-viz`)
- Evaluation logic (that is `mlpl-eval`)
- Array internals
- Interactive debugging UI
