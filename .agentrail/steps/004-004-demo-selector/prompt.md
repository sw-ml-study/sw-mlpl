Add a demo selector to the web REPL.

1. Create a demos module in apps/mlpl-web with preloaded examples:
   - "Basics" — scalar arithmetic, arrays, variables
   - "Matrix Ops" — reshape, transpose, reductions
   - "Math Functions" — exp, log, sqrt, sigmoid
   - "Logistic Regression" — full AND gate training
   Each demo is a Vec of input lines to execute sequentially.

2. Add a dropdown/select element in the control bar:
   - Label: "Load Demo"
   - Options: each demo name
   - On select: clear REPL, execute demo lines one by one,
     showing each input and its output

3. Add a "Clear" button next to the demo selector

4. Add a footer with:
   - "MLPL v0.2 — Array Programming Language for ML"
   - Link to GitHub repo
   - "Built with Rust + Yew + WASM"

5. Verify all demos execute correctly in the browser

Allowed: apps/mlpl-web/