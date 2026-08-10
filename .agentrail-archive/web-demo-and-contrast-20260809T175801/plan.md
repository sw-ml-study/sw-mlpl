# Saga: web-demo-and-contrast

Two web-UI updates that share one pages rebuild + deploy:

1. Add a public-browser demo **Rolling Retrain (LEFTS-inspired)**
   to the Experiment Quality group, adapted from
   ../demo-ml-utils/demos/experiments/lefts_page_web.mlpl (a
   leakage-safe expanding-window monthly Ridge "Lift" over a Ridge
   "Leaf"). Browser-CPU, self-contained, deterministic; no new
   builtins/syntax/types; keep def-u doc strings; trim tutorial
   prints to a few visible result lines. Add "Experiment Quality"
   to SECTION_ORDER (after Training & Learning, before Classical
   ML) + update demo_order_tests. Pass the web demo smoke gate.

2. Fix low-contrast demo-group NAMES in the web UI: the demo
   dropdown's <optgroup> labels render in the browser-default muted
   gray. Add a `.demo-select optgroup` rule (larger, bolder,
   bright accent color) so group names are legible. Theme is
   Catppuccin Mocha (dark only) -- no light theme to handle.

3. Rebuild pages + deploy (gh-pages); the demo is CPU/live so no
   serve rebuild. Report + --done.

## Steps
1. rolling-retrain-demo -- demos.toml [[demos]] record (intro,
   takeaway, adapted lines) + SECTION_ORDER + demo_order_tests;
   pass `scripts/gate.sh components/web-demos mlpl-web-demos
   mlpl-web-demos-smoke`.
2. demo-group-contrast -- `.demo-select optgroup` CSS in
   components/web/crates/mlpl-web/index.html.
3. rebuild-deploy -- build-pages.sh, commit pages/, deploy-pages.sh,
   ts-suffixed review URLs, docs/wiki if needed, --done.
