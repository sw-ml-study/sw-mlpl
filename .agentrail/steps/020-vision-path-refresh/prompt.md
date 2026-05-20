Saga 29 step inserted: refresh apps/mlpl-web/src/paths.rs Vision Transformers learning path. The path's two tail notes describe :upload x as 'still in progress' / 'coming next' even though it shipped in step 016, picked up decode-error handling in 017, and now plays nice with the generic running spinner from 018. Update the path to:

1. Intro note: drop 'still in progress as Saga 29 step 016' -- reference :upload as live tooling.
2. Replace the 'Coming next: bring-your-own-image' Note with a 'Bring-your-own-image (shipped)' Note that walks through the actual workflow: :upload x -> svg(unwrap(x).pixels, 'gallery') -> classify one-liner. Mention the four Err flavors (cancelled / decode failed / read failed / image-load failed) added by step 017.
3. Add a Demo step pointing at one of the existing pets demos as the 'try it now' anchor (the upload feature is REPL-bound, not a packaged demo, so the path needs a glue note rather than a Demo entry for :upload itself).
4. Update 'Beyond this path' to reference docs/better-cat-dog-future-demos.md (added in step 017) as the recommended improvement ladder.

While in paths.rs, also do a quick once-over of OTHER learning paths to ensure no other 'in progress' / 'coming next' phrases refer to features that have since shipped (Result type from step 012; multi-head; stack; heatmap_grid; running spinner; viz legends from step 019).

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push so the path renders correctly on the live demo.