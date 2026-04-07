Build the browser REPL UI in apps/mlpl-web using Yew.

1. Implement the main App component in src/main.rs (or src/lib.rs + main.rs):
   - Header: "MLPL v0.2" title
   - Output panel: scrollable div showing REPL history
     (input lines prefixed with "mlpl> ", results below)
   - Input area: single-line text input at bottom
   - Submit on Enter key press
   - Auto-scroll output to bottom on new content
   - GitHub corner link (SVG, top-right, links to repo)

2. REPL state management:
   - Use mlpl-wasm session API for persistent environment
   - Store history as Vec<(String, String)> (input, output)
   - :clear command resets environment and history
   - :help command shows built-in function list in output

3. Styling (Catppuccin Mocha):
   - Dark background (#1e1e2e base)
   - Monospace font throughout
   - Input has subtle border, focus highlight
   - Output text is light (#cdd6f4)
   - Error text is red (#f38ba8)
   - Responsive layout (works on mobile)

4. Basic keyboard handling:
   - Enter submits input
   - Up/Down arrow for command history navigation
   - Ctrl+L or :clear to clear

5. Test: verify trunk build succeeds and the page renders
   (manual verification is OK for UI)

Allowed: apps/mlpl-web/