Saga 23 step 006: typed-describe.

Surface ValueTags in :describe / :vars / :tags / :untag REPL commands so the educational payoff of typed values reaches the user. Today (after steps 001-005) tags exist in env.tags and flow through producer/predicate/propagation but :describe still prints shape-and-values only.

Rules to ship:

1. :describe <name> learns to consume tags. Header line shows the tag with its display_name + relevant metadata, e.g. 'logits -- Logit[batch=4, vocab=8]' or 'loss -- Loss(CrossEntropy)' or 'g -- Gradient(wrt=W1)'.

2. Per-tag body lines tailored to the variant:
   - Probability: include a one-line invariant note showing per-row sum verified-or-violated.
   - Gradient: show wrt and (if known from optim_state) which optimizer step last consumed it.
   - Weight: show layer + name + initial seed if recorded in env.
   - Activation: show producing layer + activation kind.
   - Loss: show kind.
   - LearningRate: just the scalar value.
   - Labels: show num_classes.
   - AttentionMap: shape + brief note that it renders as a heatmap via svg(_, 'heatmap').

3. :vars learns to show the tag in its one-line summary, e.g.
     'logits  Logit[batch=4, vocab=8]   shape=[4, 8]'

4. :tags REPL command -- new -- list every tagged binding sorted by tag-display-name. Shows binding name + tag header line.

5. :untag <name> REPL command -- new -- clears a tag. No-op when name has no tag. Reports 'untagged X' or 'X had no tag'.

Implementation lives in crates/mlpl-eval/src/inspect.rs (the existing :describe/:vars surface) plus the REPL command dispatch in crates/mlpl-cli (terminal) and crates/mlpl-wasm (web). Both surfaces should produce identical output for the same env state.

TDD: failing tests in crates/mlpl-eval/tests/typed_describe_tests.rs covering the per-tag rendering and the new commands.

Quality gates: full /mw-cp pass. sw-checklist failed count must hold at 139. Web UI changes require pages/ rebuild (only if mlpl-wasm output changes; if only inspect.rs changes, no rebuild needed -- the wasm build picks up via :tags command surfacing through the existing REPL dispatch).