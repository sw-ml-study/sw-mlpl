# MLPL Foundation Saga

## Quality Requirements (apply to EVERY step)

Every step MUST:
1. Follow TDD: write failing tests FIRST, then implement, then refactor
2. Pass all quality gates before committing:
   - cargo test (ALL tests pass)
   - cargo clippy --all-targets --all-features -- -D warnings (ZERO warnings)
   - cargo fmt --all (formatted)
   - markdown-checker -f "**/*.md" (if docs changed)
   - sw-checklist (project standards)
3. Update relevant docs if behavior changed
4. Use /mw-cp for checkpoint process (checks, detailed commit, push)
5. Push immediately after commit

## Goal

Bootstrap the MLPL monorepo from skeleton to a working Cargo workspace
with crate skeletons, contract prose, and agent coordination infrastructure.
This is Saga -1 and Saga 0 from the planning docs.

## Phases

### Phase 0: Repo scaffolding and agent infrastructure
- Rewrite process.md for MLPL
- Populate contract READMEs with prose specs
- Bootstrap Cargo workspace with crate skeletons

### Phase 1: Core contracts and design
- Write detailed array-contract spec
- Syntax design spike (docs/syntax-core-v1.md)
- Implement mlpl-core basics (spans, identifiers)

### Phase 2: First array types
- Shape type in mlpl-array
- DenseArray with storage and indexing
- Reshape and transpose operations

### Phase 3: Parser foundation
- Lexer for MLPL v1 syntax

## Success criteria
- cargo check succeeds for all workspace members
- cargo test passes with real tests for core, array, parser
- Contract READMEs exist with meaningful prose
- Syntax v1 spec exists
- process.md is MLPL-specific
- All code formatted, linted, pushed