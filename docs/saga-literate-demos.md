# Saga plan: literate-demos

Status: PLANNED (not yet instantiated in `.agentrail/`). Create with
`agentrail init --name literate-demos` once the active `local-gpu-agentic`
saga's pending MLX step 004 is finished (user decision, 2026-05-31).

## Vision

Convert every web-playground demo into a literate Org document whose
MLPL source blocks are live, publish each to a standalone HTML file,
bundle those HTML files with the GitHub Pages live demo, and link them
from the UI. The literate pages let the public live demo convey
use-cases it cannot itself run -- above all the connect-only demos.

Builds directly on the `ob-mlpl` `:session` support and the batch
publisher shipped in `examples/literate/` (basics.org -> basics.html).

## Load-bearing decisions (user, 2026-05-31)

- **Generator is a Rust crate that reads the demo registry.** One
  `.org` per `Demo`, emitted from `intro` (preamble) + per-line `#`
  comments (inter-step prose) + `takeaway` (conclusion). Single source
  of truth -- regenerating after a demo edit cannot drift. New crate
  lives in `components/literate/crates/` (components rule: never
  `crates/foo`).
- **Sources + outputs in `examples/literate/`, bundled to
  `pages/literate/`.** `.org` sources and generated `.html` are both
  committed under `examples/literate/`; `build-pages.sh` copies the
  `.html` into `pages/literate/` for deploy.
- **Connect-only demos are the priority, and they SHOW real output --
  clearly marked CANNED, not live.** Their `.org` carries output
  CAPTURED from a real connected run (`mlpl-serve` + Ollama / MLX),
  baked in, with `:eval no` so the batch publisher never re-runs (and
  never wipes) it. The literate page states plainly that the output is
  a recorded run, not live, and what server it requires.
- **Replace gray-out with: one explanatory REPL comment + a link.**
  Instead of rendering connect-only demos visible-but-disabled in the
  dropdown, selecting one does NOT run steps and does NOT just drop a
  bare link. It shows a SINGLE REPL comment line (no other steps)
  explaining why this cannot run live in the browser-only public demo,
  then links to the literate HTML example (the canned walkthrough). So
  the public live demo advertises the connect-mode use-cases (Ask
  Ollama, MLX LoRA fine-tune) it can't run in-browser, without
  pretending to execute them.
- **CPU demos run live in batch;** image-upload / impractically-heavy
  ones fall back to code-only with a "run locally" note.

## Shared contract

A `demo_slug(name) -> String` (kebab-case) in
`mlpl-web-demos-types`, used by BOTH the generator (output filename)
and the web UI (link target `literate/<slug>.html`), so the link and
the file can never disagree.

## Phased steps

1. **slug + runnability contract** (`mlpl-web-demos-types`). Add
   `demo_slug` and a `Runnability` classifier (live / captured-connect
   / code-only) keyed off `capability_for` + a small heuristic for
   image/heavy demos. TDD: slug stability + known classifications.
2. **generator lib** (`components/literate/crates/mlpl-literate-gen`,
   pure). `demo_to_org(&Demo, Capability) -> String`: title, `:session`
   property, per-line `#+begin_src mlpl` blocks with the trailing
   comment lifted into prose, takeaway, shared footer linking to the
   index. Non-live demos get `:eval no` + a "run it" note. Modules
   split per `code_metrics.md` (render / footer / classify / model /
   lib facade).
3. **index page** (generator lib). `index_to_org(&[Demo]) -> String`
   grouping demos by `category` with links to each `<slug>.html`.
4. **generator binary + script** (thin effects shell). Writes one
   `.org` per demo + `index.org` into `examples/literate/`;
   `scripts/gen-literate.sh` drives it. Test: 42 `.org` + index emitted.
5. **batch publish-all**. `scripts/publish-literate.sh` (+ reuse
   `examples/literate/publish.el`) globs `*.org`, executes live ones
   maintaining `:session`, leaves `:eval no` ones' baked output intact,
   exports each to `.html`. Stop git-ignoring `examples/literate/*.html`.
6. **capture connect-demo output (canned)**. Against a real
   `mlpl-serve` (Ollama on localhost/large12; MLX once step 004 lands),
   capture Ask-Ollama and MLX-LoRA output into their `.org` with
   `:eval no`, commit. The page header states it is a recorded run, not
   live, and names the server it needs.
7. **bundle into pages**. `build-pages.sh` copies
   `examples/literate/*.html` -> `pages/literate/`; commit + rebuild.
8. **UI: per-demo link + connect-demo redirect**. Where
   `Demo.intro`/`takeaway` render, add a "Literate walkthrough" link to
   `literate/<slug>.html` for every demo. For connect-only demos,
   REPLACE the gray-out: selecting one emits a single REPL comment
   explaining why it can't run live in the browser-only public demo,
   then shows the link to the canned literate example -- no step
   execution. Rebuild pages.
9. **UI: index link from help**. Link `literate/index.html` from the
   lang-ref / usage help panel. Rebuild pages.
10. **docs + close**. Update `examples/literate/README.md`,
    `docs/emacs-support.md`, `docs/language-status.md`; run
    `sw-checklist components/` and report the FAIL/WARN delta; refresh
    `CHANGES.md`. `--done`.

## Inventory at planning time

42 demos (`components/web-demos/.../aggregator.rs`). Connect-only per
`DEMO_CAPABILITIES`: **Ask Ollama (contextual)** (CPU+connect) and
**MLX LoRA fine-tune** (MLX+connect). Everything else defaults to
`Capability::CPU_LIVE`.
