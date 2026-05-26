Step 001: Editor tab + textarea.

New HeaderMode::Editor. Add Editor tab to the header (between Paths and Tour). Clicking it shows a full-height textarea with monospace font. Buttons at top: Run, Load, Save, Clear. No execution yet -- just the UI shell.

New editor_panel.rs (2 fns max): EditorPanel component with textarea + button bar. editor_state in UiState: show_editor bool + editor_content String.

Design to warning targets: <=4 fns per module. Pages rebuild required.