Tier 1 saga step 006 (final, use --done on agentrail complete): close out audit findings #18 and #19 in docs.

1. In docs/language-audit.md:
   - Finding #18 (concat axis restricted to {0, 1}): add a 'Fixed in:' line citing the commit SHA(s) from steps 001 + 002 + 003 of this saga. Keep the rest of the finding for historical context.
   - Finding #19 (multi-head attention forward-only tape): same treatment with the step 004 commit SHA. Update the migration cost paragraph to past tense ('shipped' instead of 'pure capability lift').

2. In docs/plan.md's 'Breaking-change candidates' section:
   - Move #18 and #19 from the Critical list to a new 'Shipped' subsection at the top of the section. Include one-line ship dates and links to the audit findings.

3. Refresh CHANGES.md via ./scripts/gen-changes.sh and commit as a docs-only follow-up.

Quality gates: markdown-checker on docs/language-audit.md and docs/plan.md; sw-checklist hold-or-lower. Docs-only commit, no cargo gates needed.

agentrail complete --done after committing.