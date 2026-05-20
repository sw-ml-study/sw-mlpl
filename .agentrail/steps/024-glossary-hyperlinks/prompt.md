Saga 29 step inserted: glossary hyperlinks across user-facing surfaces.

Every glossary term mentioned in a demo intro/takeaway, tutorial lesson intro, learning-path Note body, learning-path Glossary "why" text, or another glossary entry's body should be a clickable link that opens a popup (or pushes onto a side panel) with that term's full glossary definition.

Implementation notes:

1. Linking surface: extend the path_body markdown-ish renderer (apps/mlpl-web/src/path_body/) to recognize a [[term]] sigil and render it as a clickable element. Sigil form is explicit (no auto-lemma matching) so authors control which words become links and the renderer stays predictable.

2. Popup UI: a small modal overlay (or pinned side panel) that shows the matched glossary entry's body using the same renderer. Esc / click-outside dismisses. If the term is not in the glossary, render the [[term]] text literally with a small warning rather than crashing.

3. Cross-references in the glossary itself: convert mentions of one glossary term inside another entry's body to [[term]] form. Bulk pass over docs/glossary.md; manually verify the result on a few key entries (patchify, take, Stack, attention_weights, :upload).

4. Surfaces to wire up: paths.rs Note bodies + Glossary why texts; demos.rs intros and takeaways and ProgressNote bodies; lessons.rs + lessons_advanced.rs intros. Demo intros are HistoryEntry::Narration today and render as plain text; either route them through the new renderer (preferred -- consistent formatting AND clickable terms) or just allow the [[term]] sigil in those slots specifically.

5. Tests: parse_blocks + new inline tokenizer for [[...]]; render verifies the popup-target attribute is correct.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push.