Trigger correction: swap Shift+Space back to Ctrl+Space for the REPL completion popup. User reversed direction after step 045: Ctrl+Space is the right choice (IDE standard -- VS Code, IntelliJ, Emacs), not Shift+Space.

Minimal symmetric changes to the step 045 commit:

(1) apps/mlpl-web/src/completion.rs:
- Module-doc trigger paragraph: change `Shift+Space` to `Ctrl+Space`. Keep the "Tab reserved for browser nav" justification.
- `is_completion_trigger(shift_key, code)` -> `is_completion_trigger(ctrl_key, code)`. The body changes from `shift_key && code == "Space"` to `ctrl_key && code == "Space"`.
- Unit test renamed back to `trigger_predicate_only_fires_on_ctrl_space`; asserts unchanged otherwise.

(2) apps/mlpl-web/src/handlers.rs:
- The leading guard reads `e.ctrl_key()` instead of `e.shift_key()`.
- Comment block updated: trigger is Ctrl+Space; Tab still untouched (reserved for browser focus traversal). Drop the prior "Ctrl+Space collides with OS bindings" claim -- that was the failed step-045 framing.

(3) docs/glossary.md:
- "Completion popup (REPL)" entry: Shift+Space -> Ctrl+Space throughout. Rationale paragraph: Tab still reserved for browser nav, AND Ctrl+Space is the IDE standard.
- "Tab completion (REPL)" redirect: Shift+Space -> Ctrl+Space.

(4) Pages rebuild required since apps/mlpl-web changed.

(5) No README count change expected (no new glossary entries).

(6) The handle_completion name from step 045 stays.

(7) The static completion logic + popup chips component are unchanged -- another trigger swap.