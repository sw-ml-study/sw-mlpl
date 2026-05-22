Scripting saga step 007: add demos/classify.mlpl as the end-to-end example, plus a 'Scripting in MLPL' section in docs/usage.md.

demos/classify.mlpl: a script that takes an image path argument via args(), loads the image (load_images or load_preloaded), picks a model based on an optional --model flag (parsed by hand from the StrList), runs inference via a small ViT or MLP, and prints the predicted label + confidence. Exits non-zero on an unreadable image.

The demo exercises every step 001-006 surface:
- args() to read CLI args
- to_number() to parse a confidence-threshold flag if provided
- if/else to branch on the --model flag (default 'mlp', alternative 'vit')
- while or for to iterate over the loaded image batch
- print() to surface the prediction
- exit(1) on Err from load_images

Touches no runtime; pure composition.

docs/usage.md: add a 'Scripting in MLPL' section walking the four critical constructs with this demo as the worked example. Show the shell invocation 'mlpl-repl -f classify.mlpl -- my_cat.jpg --model vit' and the expected output.

TDD:
- Integration test in crates/mlpl-eval/tests/ (or apps/mlpl-repl/tests/) that spawns the binary with a known image path (a small fixture from pets_tiny, or a synthetic one) and asserts the printed-label match for a known input.
- The every_quick_web_demo_runs test in apps/mlpl-web is unchanged (this demo is CLI-only; it does not need a web playground entry).

Quality gates: cargo test workspace; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; markdown-checker on docs/usage.md; sw-checklist hold-or-lower. Push.

After this step ships there is a copy-pastable scripting starter in the demos/ tree.