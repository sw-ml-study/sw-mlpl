Saga 29 step inserted: user-facing strings in the web playground (learning paths, glossary entries shown in paths, demo intros / takeaways, ProgressNotes, REPL output messages) should not reference 'Saga 29 step N' or other developer-process metadata. The user of the web UI is a person learning ML, not a developer of this repo. 'Saga' is a dev concept tracked in agentrail; it belongs in CLAUDE.md, docs/saga.md, .agentrail/, and git commit messages -- NOT in glossary entries the user sees in the dropdown or path walker.

Concrete scope:

1. apps/mlpl-web/src/paths.rs: scan every Note.body, Glossary.why, and Demo.why for the phrase 'Saga' / 'saga'. Replace with feature-grounded descriptions ('A recent update added...', 'Earlier in MLPL...', 'The tape autograd lets...'). Where the saga number is the ONLY hook for the reader to find more, replace with concrete cross-references (a builtin name, a demo name, a glossary term).

2. docs/glossary.md: glossary entries are rendered VERBATIM in path step bodies (paths_view.rs Step::Glossary case). Audit entries used by user-facing paths -- start with the ones cited by paths.rs (patchify, take, Stack, Oxford-IIIT Pet, :upload, plus anything in the other paths). Strip saga refs; rewrite as features the user actually encounters.

3. Special rewrite: the ':upload (REPL command)' glossary entry. The current body is a wall of text full of 'Saga 29 step 016' / 'step 017' / 'step 012' refs and crams bullets inline. Rewrite as a short, scannable paragraph or two grounded in WHAT the user does and WHAT the four Err flavors mean -- no saga numbers. Use the renderer's plain-paragraph behavior (no markdown) to advantage; keep paragraphs short and flowing.

4. apps/mlpl-web/src/demos.rs: scan demo intros and takeaways for 'Saga' references and strip them. Demo body is what shows in the About panel before a demo runs; it's the most user-visible string we have.

5. apps/mlpl-web/src/help.rs help text + apps/mlpl-web/src/lessons.rs tutorial lessons: same audit.

6. ProgressNote bodies in demos.rs: audit for saga refs.

Out of scope (stays referenced): docs/*.md beyond glossary.md (architecture, plan, milestone-vit, etc.) -- those ARE developer docs and saga numbers are appropriate there. CLAUDE.md, AGENTS.md, COORDINATOR.md -- developer protocol docs. Commit messages -- forever-correct to mention saga numbers there.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist. Pages rebuild + push so the live demo reflects the user-friendly text.