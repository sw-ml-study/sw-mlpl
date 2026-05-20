Saga 29 step inserted: extend the "Decision Boundary" demo (apps/mlpl-web/src/demos.rs) to walk through AND, OR, NAND, NOR, and XOR. The current demo only trains AND. Extending the demo turns it into the classic linear-vs-nonlinear pedagogical sequence:

- AND, OR, NAND, NOR: four boolean functions that ARE linearly separable. The same 2-weight + bias logistic regression converges to clean boundaries for all four with different label vectors.
- XOR: NOT linearly separable. Train the same architecture on XOR labels and watch it stall around 50-75% accuracy with a boundary that cannot cleanly separate the four points. This is the famous Minsky/Papert result; the demo should explicitly call out the failure and point the reader at "Moons MLP" or "Tiny MLP" as the architectures that handle XOR via a hidden layer + nonlinearity.

Implementation approach:

1. Restructure the demo body so the labels (the y vector) are the only thing that changes between sections. Reuse the 4-point input X = [[0,0], [0,1], [1,0], [1,1]] across all five gates.

2. For each gate, train the same logistic regression for 300-400 steps then render the decision_boundary surface over a 20x20 grid. Use the new path-body markdown-ish renderer style (paragraphs + bullets) in the intro / takeaway so the comparison is readable.

3. The takeaway is the actual lesson: "Four out of five worked. XOR did not -- not because we trained too little, but because the architecture cannot represent it. The next demo ('Tiny MLP' or 'Moons MLP') adds a hidden layer and nonlinearity, which turns XOR into a learnable problem."

4. Keep the existing AND-only demo body intact for now (so the dropdown still has the "quick" version), OR replace it with the 5-gate version -- decide based on demo length. A long-form demo is fine if the linear-vs-nonlinear arc is the payoff. If you replace, also update apps/mlpl-web/src/paths.rs (any path that references "Decision Boundary") and any tests that assert the demo's existence.

5. Use `is_ok` / `assert_close`-style sanity checks in tests/integration smoke if the demo is too long for the workspace smoke walk.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push.