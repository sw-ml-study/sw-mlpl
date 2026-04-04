Implement the first real types in mlpl-core. This crate is the stable shared foundation.

Implement:
1. Span type -- source location tracking (start offset, end offset, optional source ID)
2. Identifier type -- validated identifier wrapper (must match syntax-core-v1 identifier rules)
3. Basic Display/Debug impls

Keep it minimal:
- No error types here (each crate owns its own errors)
- No primitive IDs yet (add later when eval needs them)
- No parser types, no array types, no runtime types

Write TDD-style:
- RED: Write tests first for Span creation, Identifier validation (valid names, rejected names)
- GREEN: Implement to pass tests
- REFACTOR: Clean up

Verify:
- cargo test -p mlpl-core passes
- cargo clippy -p mlpl-core -- -D warnings passes

Allowed directories: crates/mlpl-core/, contracts/core-contract/