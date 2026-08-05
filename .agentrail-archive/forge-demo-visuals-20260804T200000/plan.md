# Saga: forge-demo-visuals
User review 2026-08-04: two of the three Data Forge demos have no
charts and are hard to assess. Add the natural visual to each
beat, state expected output explicitly in the takeaways, and
verify the rendered SVGs directly (not just the numbers).
## Steps
1. visuals -- rejection: truth-vs-candidate scatter colored by
   the oracle (accepted points ARE the diagonal) + noise hist;
   graph: adjacency heatmaps (full/seen/unseen -- built loop-free
   via one_hot matmul) + verify bar charts (all-ones vs the
   zero-leakage all-zeros); arithmetic: confusion-matrix finale
   via predict_batch. Repl-run each, inspect the SVGs, update
   takeaways with what-you-should-see; smoke; pages+server deploy.
