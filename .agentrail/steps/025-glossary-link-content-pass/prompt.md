Saga 29 step inserted: glossary-link content pass. Step 024 shipped the [[term]] sigil + popup machinery; this step populates that machinery with actual links across the high-traffic surfaces:

1. Demo intros / takeaways (apps/mlpl-web/src/demos.rs): every demo intro has 4-8 candidate terms (Attention, softmax, embed, cross_entropy, predict_batch, etc.). Wrap ones that have glossary entries. Focus first on the 10 demos new users hit (Basics, Decision Boundary: logical gates, Pets: cat vs dog (quick), Pets: multi-head ViT (quick + viz), Tiny LM, etc.), then sweep through the rest.

2. Tutorial lesson intros (apps/mlpl-web/src/lessons.rs + lessons_advanced.rs): same pattern. Lesson intros tend to be longer than demo intros so they have more candidates per lesson.

3. Glossary cross-references (docs/glossary.md): bulk pass. For each entry body, find mentions of OTHER glossary terms and wrap them in [[term]]. Recursive linking already works (clicking a link inside a popup opens the next popup), so this turns the glossary into a navigable graph.

4. Conservatism: only link the FIRST mention of a term in a given body. Linking every occurrence is noisy.

5. Verification: spot-check 5 demos and 5 lessons in the browser after the build; click 3-4 links per to confirm they open the right popup. Click a cross-reference from within a popup to verify the recursive case.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push.

Out of scope: linking in CLI / terminal REPL output (no popup machinery there); auto-lemma matching (still explicit sigils only); a [[shortname|full term]] aliased-link syntax (small future enhancement; for now use the full glossary heading as the sigil content).