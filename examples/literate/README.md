# Literate MLPL with Org-babel

This directory holds literate-programming examples: Org documents whose
MLPL source blocks are *live*. Publishing one evaluates every block in
order, captures its output, and exports a standalone HTML file.

## Files

- `basics.org` -- recreates the web playground's **Basics** demo as a
  literate tour. Nine source blocks share one `:session basics`, so a
  variable bound in an early block is still in scope later -- the same
  state model as typing successive REPL lines. Prose between the blocks
  narrates each step.
- `publish.sh` / `publish.el` -- batch publisher. No interactive Emacs.
- `basics.html` -- generated output (git-ignored; regenerate any time).

## Publish

```bash
./examples/literate/publish.sh                       # defaults to basics.org
./examples/literate/publish.sh examples/literate/basics.org
```

The script runs `emacs -Q --batch` (no user init), loads the MLPL
Org-babel support via `elisp/mlpl-all.el` (which resolves the
`mlpl-repl` binary -- `exec-path` first, then the `sw-install`
location), resets the session, evaluates the buffer top to bottom, and
writes `basics.html` beside the source.

Requires a `mlpl-repl` binary on `PATH` or at
`~/.local/softwarewrighter/bin` (build with `cargo build -p mlpl-repl`).

## How session state works

MLPL has no long-lived interpreter process, so `ob-mlpl` models a
`:session` as the *accumulation* of every block run in it: each block
re-runs the whole accumulated program through `mlpl-repl -f` and shows
only the output its own lines added. MLPL script output is
deterministic and append-only, so the per-block delta is exact.

Re-running a buffer top-to-bottom interactively would append blocks to
the session twice; the publisher calls `org-babel-mlpl-reset-session`
first, and you can call `M-x org-babel-mlpl-reset-session` by hand
before a manual re-run.

## Write your own

````org
#+PROPERTY: header-args:mlpl :session mine :results output :exports both

First step:

#+begin_src mlpl
a = [1, 2, 3]
a
#+end_src

Later steps see `a`:

#+begin_src mlpl
a * 2
#+end_src
````

Then `./examples/literate/publish.sh path/to/your.org`.
