Saga 29 step inserted: upgrade the learning-path step renderer to handle multi-paragraph + bulleted content with proper formatting. Today apps/mlpl-web/src/paths_view.rs renders Note.body and Glossary.body as one paragraph element -- so any newline or "- " bullet in the source string becomes inline text. The :upload glossary entry and several Note bodies render as an unreadable wall of run-on text.

Concrete scope:

1. Add a small markdown-ish body renderer (new apps/mlpl-web/src/path_body.rs or inline in paths_view.rs). Parse:
   - Blank-line-separated paragraphs (consecutive newlines => separate paragraphs)
   - Lines starting with "- " or "* " => <ul><li> (group consecutive bullet lines under one list)
   - Numbered lines starting with "1. " / "2. " => <ol><li>
   - Inline backtick code spans => <code>
   - Inline **bold** => <strong>
   - Inline _emph_ => <em>
   Not a full markdown parser, just the constructs the existing body strings use.

2. apps/mlpl-web/src/paths_view.rs: replace the Note.body and Glossary body single-paragraph render with the new helper.

3. CSS in apps/mlpl-web/index.html: widen the path step panel and add max-height + overflow-y: auto so long content scrolls inside its own box. Style the new ul / ol / code / strong elements to match the playground theme.

4. Optional: apply the same renderer to lesson intros if they show similar wall-of-text issues.

Out of scope: full markdown parser (no images, links, tables, code blocks). Goal is making existing content readable.

Quality gates: cargo test / clippy / fmt / markdown-checker / sw-checklist. Pages rebuild + push.