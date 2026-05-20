Saga 29 step inserted: generic 'something is happening' spinner for the web demo runner. Today demos like 'Pets: multi-head ViT (quick + viz)' have ProgressNotes only at SOME lines (train, attn_maps render) -- the logits-post-train forward pass and the gallery svg both pause for several seconds with no visible indication that the demo is still running. User has reported this 4+ times.

Concrete changes:

1. apps/mlpl-web/src/state.rs: add EntryKind::Running variant (or a  flag on HistoryEntry) so the renderer can distinguish 'this line is being evaluated' from 'this is the result'.

2. apps/mlpl-web/src/handlers.rs: in schedule_demo_line, BEFORE the Timeout::new(0) that runs the eval, push a Running marker entry for the current line (input = the line text, output = something like 'evaluating...'), set history. The Timeout fires AFTER the browser paints, so the spinner appears. When the eval completes, pop the Running marker and push the actual result entry. The browser paints again.

3. CSS (apps/mlpl-web/index.html or styles): add a @keyframes spin + a small inline-block spinner span (~16-20px circle with a dashed border that rotates 1.2s linear infinite). CSS animations run on the compositor thread, so they continue animating during JS-blocking WASM evals.

4. Components: render the Running entry with the spinner widget next to the input line text. Faster lines (those that complete in <50ms) will see the running entry replaced too quickly to register visually; that's fine. Slow lines (train, multi-image apply) get the spinner.

5. For one-shot REPL submissions in make_submit_batch: same pattern (push running, yield via Timeout(0), eval, replace). May add line-count complexity; if too invasive for one step, scope to the demo-runner path only and note REPL submissions as a follow-up.

Tests: add a smoke test that schedule_demo_line writes a Running entry then replaces it (mock the timeout via tests::yield_now() or similar pattern).

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist (157). Pages rebuild + push.