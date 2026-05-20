Saga 29 step inserted: demo multi-part delineation. User reports the Decision Boundary: logical gates demo (and other multi-part demos) needs clearer structure: a block comment before each variant describing what that part does, "Part N of M" numbering, and a clear failure-explanation block for XOR pointing the reader at the working MLP demo.

Concrete scope:

1. apps/mlpl-web/src/demos.rs "Decision Boundary: logical gates": for each of the five gates (AND, OR, NAND, NOR, XOR), add a 3-4 line block comment before the training section that:
   - says "Part N of 5: <GATE>"
   - describes the truth-table behavior in plain English
   - notes whether the gate is linearly separable
   The XOR block additionally calls out the failure mode and names the "Decision Boundary: XOR (with MLP)" demo as the next step.

2. Add a new "XOR (not linearly separable)" glossary entry so [[XOR (not linearly separable)]] can be used as a link inside the demo comment (the demo-to-demo navigation is via the dropdown, not a click; the glossary entry explains WHY the MLP version is needed).

3. Pattern: where future demos are multi-part, follow the same "Part N of M: <NAME>" comment header convention.

4. Apply lightly to the "Pets: predict + gallery" demo since it has 6 chunked train blocks (Part 1 of 6 ... Part 6 of 6) -- those already exist conceptually; just add the headers.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push.