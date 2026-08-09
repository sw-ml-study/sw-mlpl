# Saga: non-browser-demos

Add a new demo category (Non-Browser / Companion CLIs) to
demos.toml: one pointer-demo per ../demo-* companion repo. Each
"demo" is comment-only (states it is NOT runnable in the
browser), links to the GitHub repo, and describes what the CLI
repo does + its app-level demos. Covers all seven: algorithms,
combinators, extensions, file-processing, functional-pipelines,
memory, ml-utils. Pins: README demo count + demo_order group
count test.

## Steps
1. add-group -- new [[demos]] entries (Non-Browser category);
   def-comments gate n/a (no defs); README count; demo_order +
   registry tests green.
2. close -- rebuild pages, deploy, --done.
