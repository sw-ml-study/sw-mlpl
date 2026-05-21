Tier 1 saga step 005: rewrite the multi-head pets demo intros and takeaways now that the multi-head model actually trains.

Affected demos:
- apps/mlpl-web/src/demos.rs: 'Pets: multi-head ViT (quick + viz)' (~line 949)
- apps/mlpl-web/src/demos.rs: 'Pets: attention overlay (per-head)' (~line 1005)
- demos/vit_multihead_quick.mlpl (the CLI version that vit_attention_viz.mlpl is built on)

Today's takeaways are calibrated against the plateau-around-0.5 reality. After step 004 the loss drops materially and accuracy reaches > 0.8 on the balanced training set. Update each demo's:
- intro: drop any 'compare with the untrained multi-head pattern to see what specialization gradient descent buys -- the heads start identical and end different' framing language that worked AROUND the lack of real training. Keep the framing about post-training specialization but make it about the now-real loss curve.
- takeaway: state the new accuracy expectation; rewrite the 'common post-training patterns' paragraph to reflect that the heads now diverge from training rather than from random initialization differences.

Also run the affected demos end-to-end via release mlpl-repl -f and capture the actual loss-curve / accuracy / attention-grid visuals -- they will look qualitatively different from before. Use playwright if needed to verify the live deployed demos show the new behavior.

Quality gates: cargo test -p mlpl-web --release (the every_quick_web_demo_runs test exercises the demo lines); cargo clippy / fmt / sw-checklist hold-or-lower; rebuild pages (apps/mlpl-web/ changed); commit + push.