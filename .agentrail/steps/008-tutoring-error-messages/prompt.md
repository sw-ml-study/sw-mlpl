Saga 23 step 008: tutoring-error-messages.

Polish the tutoring hint catalog. Step 004 (predicate consumers) shipped 5 hint constants for the Logit and Loss consumer predicates. Step 005 added one for domain-mismatch arithmetic. Step 008 polishes them as the load-bearing educational deliverable: every type-related error should be a 3-5 line tutor naming the cause and a concrete fix.

Audit and improve:
1. Review the existing hint constants in crates/mlpl-eval/src/type_errors.rs and crates/mlpl-eval/src/tag_propagate.rs against actual MLPL syntax (e.g. softmax takes an axis, log_softmax isn't a builtin yet). Each hint should reference real MLPL code paths the user can copy-paste.
2. Add hints for any uncovered (op, got_tag) pairs:
   - sample/top_k vs LogProbability (if not already)
   - cross_entropy vs Labels-tagged first arg (a common confusion: passed labels instead of logits)
3. Each hint should follow a consistent shape: 1) what the op expected and why, 2) what it got, 3) likely cause, 4) one or two concrete fixes with copy-pasteable MLPL syntax.
4. Add a contracts/eval-contract/tutoring-hints.md catalog so hints can be reviewed in one place and so trace consumers / future docs can render them.

TDD: failing tests in crates/mlpl-eval/tests/tutoring_hints_tests.rs that assert specific phrasings (canonical fix-MLPL strings) appear in the hint output. The tests double as a contract check that hints don't drift away from valid MLPL syntax.

Quality gates: full /mw-cp pass + markdown-checker on the new contract doc. sw-checklist failed count must hold at 139.