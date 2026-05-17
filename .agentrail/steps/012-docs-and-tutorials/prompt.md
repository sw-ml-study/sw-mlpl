Saga 21.5 step 012: docs-and-tutorials.

Goal: retrospective docs sweep for everything Saga 21.5 shipped. Move every Phase 1-6 deferred item out of the "Non-goals (deferred)" list in docs/using-cli-server.md into the main body. Add two web REPL tutorial lessons in apps/mlpl-web/src/lessons.rs: "Connect to a remote MLPL server" and "Long training, live loss." Update docs/saga.md Saga 21.5 entry, docs/status.md one-liner. Rebuild pages/.

TDD:

1. RED tests: minimal. docs/using-cli-server.md and the tutorial lessons are user-facing prose; the existing readme_counts test in apps/mlpl-web/src/readme_counts.rs may need its lesson-count constant bumped if I add lessons. Add a test that counts the number of "Saga 21.5" entries in docs/saga.md is >= 1.

2. GREEN:
   - docs/using-cli-server.md: move SSE streaming + cancellation + viz storage URL + web REPL connect mode + session persistence + f32/u8 wire dtype out of "Non-goals". Add a Streaming and cancellation section. Add a Web UI in connect mode section. Add a Session persistence section.
   - apps/mlpl-web/src/lessons.rs: two new lesson entries.
   - apps/mlpl-web/src/readme_counts.rs: bump the expected counts if tests assert on them.
   - docs/saga.md: add a Saga 21.5 narrative entry summarizing what shipped across steps 001-011.
   - docs/status.md: update the Saga 21.5 row to v0.20.0 [x] (after step 013) or in-progress.

3. Rebuild pages/ after the lesson additions.

Quality gates per /mw-cp: cargo test, cargo clippy, cargo fmt, markdown-checker, sw-checklist (held). scripts/build-pages.sh.

Out of scope: release v0.20.0 (step 013); the full Environment serde across mlpl-eval; the orchestrator-side dtype picker.