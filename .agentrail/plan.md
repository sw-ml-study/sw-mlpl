# REPL to Script Learning Path

Proposed milestone (saga TBD, after UDF saga).

## Why this exists

MLPL has two execution modes: interactive REPL and script
files (.mlpl via `mlpl-repl -f file.mlpl`). The web
playground adds a third: the Editor tab with Run/Load/Save.
But there is no guided walk that teaches a learner to
graduate from one-line REPL exploration to multi-line
scripts to saved .mlpl files.

## Goals

- **Progression.** REPL exploration -> multi-line editor ->
  .mlpl file -> script with arguments.
- **Practical.** Each step produces something the learner
  keeps -- a saved script, a workflow, a habit.
- **Connects to UDFs.** Once `def u:name(args) { body }`
  lands (saga 46), scripts become reusable libraries.

## Learning path steps (proposed)

1. **Note: "From exploration to automation"** -- framing.
   The REPL is for exploring; scripts are for repeating.

2. **Lesson: "REPL basics"** -- existing "Hello Numbers"
   and "Variables" lessons. Variables persist across lines;
   `:vars` shows what is bound; `:clear` resets.

3. **Lesson: "Multi-line in the editor"** -- the Editor tab.
   Type multiple lines, Ctrl+Enter to run. Output appears
   in the REPL pane. Copy/paste between REPL and editor.

4. **Note: "Saving your work"** -- Save button downloads
   a .mlpl file. Load button reads one back. The browser
   does not persist state -- save early, save often.

5. **Demo: "Script workflow"** -- a short .mlpl script
   that loads data, trains a model, and prints results.
   Shows the pattern: setup, train, evaluate, report.

6. **Lesson: "Running scripts from the terminal"** --
   `mlpl-repl -f my_script.mlpl`. Stdin piping splits
   `repeat {}` blocks (known limitation); always use `-f`.

7. **Note: "Script arguments"** -- `args()` and
   `list_get(args(), 0)` for parameterized scripts.
   `mlpl-repl -f script.mlpl -- arg1 arg2`.

8. **Note: "Toward functions"** -- teaser for UDFs.
   Once `def u:name(args) { body }` lands, scripts
   become reusable libraries. The `u:` namespace keeps
   user functions separate from builtins.

## Dependencies

- Saga 46 (UDFs) should land first so step 8 has
  runnable content instead of a "coming soon" note.
- Editor tab (saga 36) already exists.
- Script mode (mlpl-repl -f) already exists.

## Quality requirements

Same as all sagas. Pure content -- no new builtins needed.
