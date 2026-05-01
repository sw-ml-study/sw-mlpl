# Inspectable ComputationGraph Milestone (Saga 25)

## Why this exists

Saga 9 built a reverse-mode autograd tape inside `mlpl-autograd`,
but the tape is internal: a student running `grad(loss, W)` sees
the gradient value and nothing else. The forward pass, the
intermediate activations, the dependency edges, and the backward
flow are invisible. `docs/research3.txt` calls this out explicitly:
the language's organizing principle should be the *observable
transform* loop:

```
value_before -> operation -> value_after -> shape change
             -> graph change -> visual animation -> gradient consequence
```

Saga 25 surfaces the tape as a first-class `Value::Graph` with
inspectable nodes, edges, forward values, and backward gradients.
It pairs with a new `svg(graph, "compute_graph")` viz type and an
`animate(graph)` builtin that renders the forward and backward
pass as a sequence of frames. Builds directly on Saga 23's typed
values: every node in the graph is a typed value, so the
visualization labels every wire with its `Logit` /
`Probability` / `Loss` / `Activation` tag.

Goal ranking applied:

- **Educational** is the headline goal. A student should be able
  to scrub through a forward pass, watch each shape transform,
  see where each gradient flows back, and read the
  `Logit -> Probability -> Loss` type narrative as a graph
  annotation.
- **Correctness** is served by exposing the graph: bugs in
  user-written autodiff (when those land) become inspectable
  rather than mysterious.
- **Performance** is explicitly not a goal. The graph value is a
  snapshot, not a live structure; rendering it walks the entire
  tape; nothing about this saga is on a hot path.

## Non-goals

- Symbolic differentiation. Tapes are concrete forward executions
  with backward formulas, not symbolic expressions.
- Graph-level optimization (operator fusion, common-subexpression
  elimination). Out of scope; possibly a future compile-time saga.
- Higher-order derivatives beyond a second-order Hessian helper
  (`hessian(f, x)`). Third-order and beyond are out of scope.
- Editing the graph after construction. The graph is read-only.
- Distributed graphs (cross-host tape). Out of scope.
- Replacing the existing `grad` builtin. `grad` continues to
  return a gradient value; the new `compute_graph(loss)` builtin
  returns the *graph* the gradient came from.

## Quality requirements (every step)

Identical to Saga 23.

## What already exists

- `mlpl-autograd` reverse-mode tape (Saga 9). Internal, but the
  data structures the new `Value::Graph` exposes are exactly the
  tape's existing nodes and edges.
- `grad(loss, w)` builtin (Saga 9) -- the precedent for taking a
  loss expression as an argument.
- `mlpl-viz` SVG framework (Saga 7) and the existing `svg(...)`
  diagram-type dispatch.
- Saga 23 typed values: every node in the graph already carries
  the right tag without further wiring.
- The `:describe` metadata pipeline (Saga 11 + Saga 23) -- ready
  to render a typed graph summary.

## Phases

### Phase 1: ComputationGraph value variant

- New `Value::Graph(GraphSpec)` variant in `mlpl-eval`.
- `GraphSpec` in a new `crates/mlpl-autograd/src/graph_value.rs`:
  - Nodes: each carries a forward value, a tag (Saga 23), an
    op name, an originating source span (where available), and
    a list of input node ids.
  - Edges: `from -> to` with the gradient that flowed across
    them on the most recent backward pass, if any.
  - Forward and backward order: two `Vec<NodeId>` defining the
    topological order the tape was executed in.
- `compute_graph(loss)` builtin returns a `GraphSpec` snapshot
  of the loss expression, after running both forward and
  backward.
- `:describe g` for a graph prints node count, edge count,
  forward depth, and the tag distribution at the leaves.

### Phase 2: Graph introspection builtins

- `nodes(g)` -- returns a string vector of node descriptions
  (op name + tag + shape).
- `edges(g)` -- returns a rank-2 array of `[from, to]` indices
  (suitable for `svg(_, "graph")`).
- `forward_values(g)` -- returns a list of forward-pass values,
  in topological order.
- `backward_values(g)` -- returns a list of backward gradients,
  in reverse-topological order.
- `node_tag(g, i)` -- returns the typed tag at node `i`.

### Phase 3: Static graph SVG

- New `svg(g, "compute_graph")` diagram type in `mlpl-viz`.
- Renders nodes as boxes labeled with op name + typed shape,
  edges as arrows labeled with the gradient magnitude (when
  backward has run). Forward edges use one color, backward
  edges another; the legend explains both.
- Layout: deterministic top-down topological layout. No
  `dagre`-style optimization; coordinates derive from depth
  and breadth-first ordering.
- The output is a static SVG suitable for inline rendering in
  the web REPL and for `--svg-out` saving.

### Phase 4: Animated graph rendering

- New `animate(g)` builtin returns a sequence of SVG frames
  walking the forward pass node-by-node, then the backward
  pass node-by-node. Each frame highlights the *current* node
  and the values flowing through it.
- Frame count = forward node count + backward node count;
  configurable via `animate(g, fps=N)` if needed.
- The web REPL learns to render an animation as a small
  `<svg>` carousel with prev / next / play controls.
- The terminal CLI viz cache (Saga 21) writes the animation as
  a sequence of files; the cache key includes the frame index.
- An optional `animate(g, kind="forward_only")` /
  `animate(g, kind="backward_only")` argument restricts the
  animation to one phase.

### Phase 5: jacobian and hessian

Closed-form helpers built on top of the graph value:

- `jacobian(loss, x)` -- returns a rank-2 Jacobian by re-running
  backward over each output dimension. Uses the graph value
  internally.
- `hessian(loss, x)` -- returns the rank-2 Hessian via
  `jacobian` of `grad`. Used by curvature-aware demos.
- Both raise a clean error when the loss is not differentiable
  with respect to `x` (no graph edge connects them).

### Phase 6: Demos

- `demos/compute_graph_basics.mlpl` -- tiny linear regression,
  `compute_graph` + `svg(g, "compute_graph")` to show the
  `X @ W + b -> mse` pipeline as a typed graph.
- `demos/animate_softmax_xent.mlpl` -- the canonical
  `Logit -> softmax -> Probability -> cross_entropy -> Loss`
  pipeline rendered as a forward + backward animation. The
  highlight demo of the saga; the type narrative is the
  pedagogy.
- `demos/hessian_curvature.mlpl` -- compute the Hessian of a
  small quadratic loss, plot eigenvalues, illustrate ill-
  conditioning.

### Phase 7: Tutorial lessons + retrospective + release

- Two new web REPL lessons:
  - "Computation Graphs" -- build a graph, inspect it, render
    a static SVG.
  - "Animating Forward and Backward" -- step through the
    softmax + cross_entropy pipeline frame by frame.
- `docs/using-compute-graph.md` retrospective + user guide.
- Update `docs/saga.md`, `docs/status.md`,
  `docs/are-we-driven-yet.md`.
- Bump REPL banners; rebuild `pages/`; tag the release.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | graph-value-variant       | 1 | `Value::Graph` + `compute_graph(loss)` + `:describe` |
| 002 | graph-introspection       | 2 | `nodes`/`edges`/`forward_values`/`backward_values`/`node_tag` |
| 003 | static-graph-svg          | 3 | `svg(g, "compute_graph")` |
| 004 | animate-graph             | 4 | `animate(g)` + REPL carousel + viz-cache integration |
| 005 | jacobian-hessian          | 5 | `jacobian(loss, x)` + `hessian(loss, x)` |
| 006 | compute-graph-demos       | 6 | three demos including the headline animation |
| 007 | compute-graph-tutorials   | 7 | two new web REPL lessons |
| 008 | compute-graph-release     | 7 | docs, banners, pages rebuild, release tag |

Eight steps. The animation step (004) is the largest; budget for
it slipping to two steps if the carousel UX needs iteration.

## Success criteria

- `g = compute_graph(cross_entropy(softmax(apply(mdl, X)), Y))`
  returns a graph value whose `:describe` reports the right node
  and edge counts.
- `svg(g, "compute_graph")` renders a top-down typed graph with
  every wire labeled by its Saga 23 tag.
- `animate(g)` produces a working carousel in the web REPL that
  walks forward through the pipeline then backward through the
  gradients.
- `jacobian(mse(apply(mdl, X), Y), W)` matches finite-difference
  Jacobian within 1e-4.
- `hessian(0.5 * sum(x * x), x)` returns the identity matrix
  within 1e-6.
- `demos/animate_softmax_xent.mlpl` renders end-to-end in the
  browser.
- All existing demos still pass.
- Quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Tape size.** A 200-step training loop produces a small graph
  per step but does not retain the whole training tape.
  `compute_graph` snapshots a *single* forward+backward; the
  saga is not about visualizing whole training runs. Document
  the limit.
- **Source spans.** The tape today does not retain source spans
  for every op; backfilling spans is a span-threading subtask.
  If span coverage is incomplete, `nodes(g)` should report
  "unknown span" cleanly rather than panic.
- **MLX graphs.** Saga 14's MLX dispatch re-materializes the CPU
  tape; the graph value should report which nodes ran on which
  device. Out of scope for the first version (CPU graphs only),
  but the schema must allow it.
- **Animation file count for the CLI viz cache.** A 50-node
  pipeline produces 100 SVG frames; the cache fills quickly.
  Mitigation: per-graph animation directories named by the
  cache key prefix; an `:animations` REPL command for cleanup.
- **Web REPL bundle size.** Rendering a 50-frame carousel as
  inline SVG is ~200KB minified. Acceptable for the demo, not
  acceptable for every graph in a tutorial. The `animate` call
  must be explicit; nothing auto-animates.
