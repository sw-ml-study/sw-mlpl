Scripting saga step 008 (FINAL, use --done): close out audit findings #22-#30.

1. In docs/language-audit.md:
   - #22 (no if/else): add 'Status: SHIPPED in saga 31 step 004 (commit X)' header.
   - #23 (no while/break/continue): SHIPPED in saga 31 step 005 (commit X).
   - #24 (no CLI args): SHIPPED in saga 31 step 003 (commit X).
   - #25 (no env()): SHIPPED in saga 31 step 002 (commit X).
   - #26 (no string-to-number): SHIPPED in saga 31 step 002 (commit X).
   - #27 (no stdin): SHIPPED in saga 31 step 006 (commit X).
   - #28 (no print): SHIPPED in saga 31 step 001 (commit X).
   - #29 (no script exit code): SHIPPED in saga 31 step 006 (commit X).
   - #30 (no script-mode example demo): SHIPPED in saga 31 step 007 (commit X).
   Keep each finding's original text for historical context.

2. In docs/plan.md's 'Breaking-change candidates' section:
   - Move all nine findings (#22 through #30) into the Shipped subsection. The Critical list collapses to #1 + #2 + #3 + #10 + #12.

3. Refresh CHANGES.md via ./scripts/gen-changes.sh and commit as a docs-only follow-up.

4. Update docs/language-status.md:
   - Clear the active-saga row.
   - Mark scripting-cluster (saga 31) shipped in the timeline.
   - Add a new entry at the top of the Shipped log.
   - Flip all nine #22-#30 rows to shipped in the per-finding table.

Quality gates: markdown-checker on the three docs; sw-checklist hold-or-lower. Docs-only commit.

agentrail complete --done.