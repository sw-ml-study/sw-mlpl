Step 006 (FINAL, use --done): Overlay lifecycle + CSS polish + saga close.

Extract overlay rendering into render_shell_overlays.rs if not already done. Single place for overlay priority and z-index stacking. Lifecycle rules: first visit -> splash; splash Tour -> tour starts; splash Dismiss -> nothing; return same version -> nothing; return new version -> what's-new; Tour header button -> tour unconditionally; Escape -> close topmost.

Final CSS polish: Catppuccin Mocha colors, responsive splash cards, tour tooltip arrow, keyboard accessibility, aria attributes.

Rebuild pages/ and commit. Final sw-checklist pass. Update docs/language-status.md. Mark saga complete with --done.