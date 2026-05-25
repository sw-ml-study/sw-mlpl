Arrow-key navigation for the REPL completion popup. Currently (after step 046) the popup renders a row of chips that require a mouse click to pick. Add keyboard navigation per the canonical IDE pattern:

(1) Add `completion_selected: UseStateHandle<usize>` to UiState alongside `completion_candidates`. Resets to 0 each time new candidates land.

(2) make_keydown logic: introduce a "popup open" predicate (`!completion_candidates.is_empty()`). When popup is open, these keys are intercepted (preventDefault):
- ArrowDown: increment selected (wrap to 0 at end)
- ArrowUp: decrement selected (wrap to last at 0)
- Enter: apply the highlighted candidate at the cursor (same insertion logic as the chip click in step 043's make_pick_completion); clear popup
- ArrowRight: ONLY when the cursor is at the end of the input AND the popup is open, apply the highlighted candidate; otherwise let ArrowRight pass through as a normal cursor move
- Escape: clear popup, keep input value as-is

When the popup is closed:
- ArrowUp/ArrowDown: existing behavior (command history -- unchanged)
- Other keys: existing behavior

The ordering matters: pop-open arms must short-circuit BEFORE the existing history-navigation arms.

(3) Update InputRow to receive the selected_index and apply a `completion-chip selected` class to the highlighted chip. Add CSS for the `.selected` state in index.html (subtle background + brighter border).

(4) Move the "apply candidate at cursor" path from make_pick_completion (currently in render_shell.rs) into a small helper in completion.rs so the keydown handler and the click handler share it. Or keep separate -- pick the cleaner shape; document either way.

(5) Update the 'Completion popup (REPL)' glossary entry: list every keybinding the popup responds to (ArrowUp/Down to navigate, Enter or ArrowRight to accept, Escape to dismiss, click on a chip also accepts).

(6) Add unit tests (pure):
- `next_index(cur, len)` and `prev_index(cur, len)` wrap-around helpers
- Predicate that says "ArrowRight should accept" requires (popup_open, cursor_at_end)

(7) Pages rebuild required since apps/mlpl-web changed.