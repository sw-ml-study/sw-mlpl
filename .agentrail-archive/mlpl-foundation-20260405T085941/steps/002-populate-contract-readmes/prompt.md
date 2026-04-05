Populate contract README.md files with meaningful prose specs. Each contract directory under contracts/ currently has only a stub README. Fill them in with:

For each contract (array-contract, parser-contract, runtime-contract, eval-contract, trace-contract, viz-contract, core-contract, ml-contract):
- Purpose (1-2 sentences)
- Key types and concepts
- Invariants and rules
- Error cases
- Open questions (if any)
- What this contract does NOT cover

Focus most effort on array-contract and parser-contract since those are needed soonest. Others can be shorter stubs with purpose and scope only.

Key design decisions to incorporate:
- Parser and array are peers (parser does NOT depend on array)
- Parser AST uses parser-owned nodes for array literals, not mlpl-array types
- Each crate owns its own error types (not centralized in core)
- mlpl-core holds only: spans, identifiers, maybe symbol/name wrappers, small shared utility types

Reference docs/architecture.md and docs/clarifications.txt for decisions.