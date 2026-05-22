Tech-debt saga step 007: warning long-tail sweep.

By this step the FAIL column should be in the 70-90 range. Pivot to warnings.

1. List the largest warning categories:
   sw-checklist -v 2>&1 | grep WARN | sed -E 's/.*WARN[^]]*\] ([A-Za-z ]+) \[.*/\1/' | sort | uniq -c | sort -rn | head

2. For each big category (Function LOC, File LOC, Module Function Count):
   - Walk the 26-30 line Function-LOC warnings first; each compress = -1 warning. Use real refactors (extract helpers) not struct-literal compressions.
   - File-LOC warnings (350-500 lines) -- split per code_metrics.md when the file has 2+ responsibilities.
   - Module Function Count warnings (>4, max 7) -- extract a helper module when 5+ fns share a clear responsibility.

3. AVOID compressions that rustfmt will revert (multi-field struct literals on one line, format!() calls that span past 100 chars, long chained method calls).

Target retirement: -50 warnings, -3-5 FAILs.

Strict gate: sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1.