Saga 23 step 009: typed-values-tutorial.

Ship the web REPL tutorial lesson 'Typed ML Values' that teaches the typing layer end-to-end. Educational-first: this is where a student first encounters Logit / Probability / Loss / Gradient / Weight tags and learns why they prevent the canonical ML bugs.

Lesson structure (markdown + MLPL code blocks; renders in the web REPL tutorial panel):

1. Section 'Untyped baseline' -- show a small classifier program with no tags. Output of :describe and :vars looks shape-only.

2. Section 'Producer auto-tagging' -- the same program, but now with a softmax + cross_entropy line. :describe shows probs as Probability and loss as Loss(CrossEntropy) with the row-sum invariant verified.

3. Section 'The canonical bug: double softmax' -- attempt cross_entropy(softmax(L, 1), Y). Show the TypeMismatch with the tutoring hint. Walk through the fix: pass the original Logit, not the post-softmax Probability.

4. Section 'Tag propagation' -- show A + B with tagged arguments propagating their tag. Then L + P (Logit + Probability) raising a domain-mismatch hint.

5. Section ':tags and :untag' -- inspect the tag side-table; clear a tag deliberately when the auto-tagger guessed wrong.

6. Section 'Typed traces' -- enable :trace, run a softmax + cross_entropy pipeline, show the trace JSON with output_type fields.

Implementation:
- New lesson file in apps/mlpl-web/src/lessons/ (or wherever the existing tutorial lessons live -- check for the Saga 11 'Model Composition' or Saga 16 'Embedding exploration' lesson as a template).
- Each section has runnable MLPL code so a student can step through interactively.
- Updates pages/ via scripts/build-pages.sh per the live-demo deploy gate (CLAUDE.md).

TDD: a smoke test in crates/mlpl-wasm/tests (or wherever the lesson testing lives) that the lesson MLPL programs run without panic.

Quality gates: full /mw-cp pass + markdown-checker on lesson markdown. sw-checklist failed count must hold at 139. Web UI changes require pages/ rebuild.