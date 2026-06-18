//! Test-only harness crate for the web demo registry. Carries no library
//! code -- it exists so the heavy interpreter dependency (mlpl-eval +
//! mlpl-parser) needed to actually RUN every demo lives here, in `tests/`,
//! instead of in `mlpl-web-demos`. Keeping it out of the facade lets the
//! light metadata tests there compile without the interpreter. See
//! `docs/modular-build.md`.
