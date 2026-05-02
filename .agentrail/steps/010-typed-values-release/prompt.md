Saga 23 step 010: typed-values-release.

Wrap Saga 23 with the user-facing retrospective doc + status / saga / are-we-driven-yet updates + version bump + REPL banners + tag the release.

Deliverables:

1. New docs/using-typed-values.md retrospective covering:
   - The Tier A vocabulary shipped in step 001 (Logit, Probability, LogProbability, Loss, Gradient, Weight, Bias, Activation, LearningRate, Labels, AttentionMap)
   - Auto-tagging rules from steps 002 and 003 (FnCall producers + model_dispatch param tagging + apply structural-tail rule)
   - Predicate consumers from step 004 with tutoring hints
   - Tag propagation table from step 005
   - :describe / :vars / :tags / :untag from step 006
   - Typed trace events from step 007
   - Polished hint catalog from step 008
   - Tutorial lesson from step 009
   Each section: what the user should know, an example, what is deferred. Keep the doc <500 lines so sw-checklist file-LOC stays clean for docs.

2. Update docs/saga.md: mark Saga 23 COMPLETE with a paragraph summarizing what shipped (matching the pattern of prior COMPLETE entries).

3. Update docs/status.md: move Saga 23 from Planned to Completed table.

4. Update docs/are-we-driven-yet.md: move Tier A typed-value rows from PLAN/CONS to HAVE.

5. Bump REPL banners. Search for the version string and bump (the project is at v0.18.0; this saga lands as v0.19.0 since the new typed-value surface is a major feature).

6. Update Cargo.toml workspace version to 0.19.0.

7. Push and tag the release: git tag v0.19.0-typed-values.

Quality gates: full /mw-cp pass + markdown-checker on the new retrospective. sw-checklist failed count must hold at 139. Web UI / banner changes require pages/ rebuild.

If this is the last step of the saga, agentrail complete --done.