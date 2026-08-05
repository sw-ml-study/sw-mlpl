# Generation state and the KV cache: design

Status: DRAFT for review (saga gen-state-kv-cache, step 001).

## The problem

Today's generation idiom recomputes the full prefix every token:

```text
repeat 20 { logits = apply(model, seq); last = last_row(logits);
            nxt = sample(top_k(last, 20), 0.8, step);
            seq = concat(seq, nxt) }
```

Each step re-projects Q/K/V for every prior position and
re-attends over the whole [T, T] score matrix, so a T-token
generation costs O(T^3) attention work. The fix every serving
stack uses: cache each layer's K and V rows, and per new token
project ONE row and attend one query against the cache --
O(T^2) total, with bit-identical outputs on CPU (the per-row
arithmetic is the same operations in the same order).

## Design constraints

1. **The loop stays visible.** Generation is a loop over TIME --
   the kind the language deliberately keeps (Thinking in
   Arrays). We accelerate the body, not hide the loop: no
   monolithic `generate()` builtin in this saga.
2. **Explicit state, explicit invalidation.** Speculative
   decoding (the next saga) needs to clone, verify against, and
   roll back generation state -- so the state must be a
   first-class value the user holds, not hidden inside a model.
3. **Exactness is the exit criterion.** Cached greedy generation
   must produce BIT-IDENTICAL token ids to the recompute path on
   CPU, pinned by tests. (MLX matches within fp32 tolerance, as
   established at the E4 seam.)

## The surface (six small builtins, `gen_` family)

| Builtin | Semantics |
|---|---|
| `gen_state(model, prompt)` | Build a GenerationState: run the prompt once, cache every attention layer's K/V rows. Returns the state value. |
| `gen_logits(gs)` | The logits row for the NEXT position (what `last_row(apply(m, seq))` gives today), from the cache -- no recompute. |
| `gen_append(gs, ids)` | Feed accepted token(s) into the state: project their rows, append to every layer's K/V. Rank-0 id or rank-1 id vector -- the multi-row form is the batched VERIFICATION hook the MTP saga builds on (k proposed tokens, one forward). |
| `gen_clone(gs)` | Independent copy (speculation branches; compare-two-continuations experiments). |
| `gen_reset(gs)` | Drop cached rows back to the prompt. |
| `gen_stats(gs)` | Record: `{tokens, layers, kv_rows, kv_values}` -- cache accounting, observable like the E4 seam counters. |

The cached loop then reads:

```text
gs = gen_state(model, prompt)
repeat 20 { nxt = sample(top_k(gen_logits(gs), 20), 0.8, step);
            gen_append(gs, nxt) ; seq = concat(seq, nxt) }
```

Same shape, same pedagogy, one complexity class faster.

## Semantics decisions

- **GenerationState is a first-class value** (a tenth value
  kind), bound like tokenizers and models; `:describe gs` prints
  the accounting record.
- **Weights are read live; the cache is not auto-invalidated.**
  Training the model after `gen_state` leaves stale K/V rows;
  that is the user's contract, and the documented idiom is
  `gen_reset` (or a fresh `gen_state`) after any optimizer step.
  Auto-invalidation would couple the optimizer to generation
  state for a case that always indicates a user bug.
- **Supported layer set** = the shipped LM chain surface: embed,
  `causal_attention`, rms_norm, linear, activation layers,
  residual, engram (hash-addressed -- naturally per-token).
  Non-causal `attention` inside a gen chain gets a tutoring
  error (its output for position t depends on future positions;
  caching cannot be exact). Anything else unsupported errors by
  name.

## Saga steps

1. **design** (this document; pause for review)
2. **gen-state-core** -- GenerationState value + `gen_state` /
   `gen_logits` / `gen_append` (single-token) on CPU, TDD; the
   equivalence test pins cached greedy == recompute greedy,
   bit-identical, over the Tiny LM chain.
3. **gen-controls** -- `gen_clone` / `gen_reset` / `gen_stats`,
   multi-row `gen_append` (the verification hook), tutoring
   errors for unsupported chains, `:describe` support.
4. **bench-and-demo** -- docs/benchmarks.md wall-clock table
   (T = 32 / 128 / 512, cached vs recompute) and a "KV Cache"
   demo: same prompt generated both ways, ids compared equal,
   gen_stats growth shown; visual = per-step cost curve (flat
   cached vs linearly growing recompute, measured in attended
   positions -- deterministic, so it renders in the browser).
5. **mlx-resident-kv** -- K/V rows as TensorHandles on the E4
   seam (dev_concat already exists from E5); equivalence within
   fp32 tolerance; crossover measured (expect wins at the same
   d~128 boundary as training).
6. **close** -- docs, queue advance (next: mtp-training), wiki.

## Open questions for review

1. Surface: the explicit `gen_*` primitive family (proposed), vs
   also shipping a convenience `generate(model, prompt, n, temp,
   seed)` wrapper now? (Proposed: primitives only; the wrapper
   can come with the speculation saga where its internals get
   interesting.)
2. Cache-vs-training semantics: explicit `gen_reset` contract
   (proposed) vs auto-invalidating the cache when an optimizer
   touches the model's params?
3. MLX residency in this saga (step 5, proposed per the queue)
   vs deferring it to ride the MTP saga?
