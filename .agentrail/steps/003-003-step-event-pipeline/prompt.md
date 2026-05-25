Phase 2: Step event pipeline.

New viz3d_events.rs module: Stage3dEvent + ShapeInfo structs (pure data, 2 items). Add event emission to the eval pipeline: after each REPL eval or demo line, emit a Stage3dEvent via window.__stage3d_add_step(json). Include step_idx, operation label, input shapes, output shape.

JS side (stage3d.js): receive events, place a labeled colored BoxGeometry at (step_idx * spacing, 0, 0). Color encodes operation type (blue=matmul, green=activation, red=loss, gray=assignment). TextSprite label shows 'op [shape] -> [shape]'. Camera auto-pans to latest step.

Test: unit tests for Stage3dEvent serialization. Pages rebuild required.