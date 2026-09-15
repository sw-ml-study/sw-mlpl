# demo-coding-agent-findings

Address the five findings from ../demo-coding-agent (CA1-CA5 in
docs/sw-mlpl-findings.md; that repo files them as its own F1-F5). Lead with CA2,
the meta-fix: an undefined-function call must say "unknown function", not the
array diagnostic that misleads every wrong builtin-name guess.

## Steps (priority order)

1. CA2 -- calling an undefined function reports the array diagnostic instead of
   "unknown function: NAME". Fix the fncall dispatch error path to name the
   real problem. TDD.
2. CA5 -- len rejects string lists. Make len accept a StrList (item count),
   keeping list_len as an alias; a single-string len errors clearly (directing
   to len_bytes/len_chars). TDD.
3. CA1 -- "a" + "b" fails with the array diagnostic. Either concatenate two
   strings with +, or give a clear "use str_concat" error. TDD.
4. CA4 -- write_text does not create parent dirs. Add a sandboxed make_dir
   builtin (like the other fs builtins) or have write_text create parents. TDD.
5. CA3 -- doc-only: a symlink whose target is inside the sandbox reads fine, but
   the docs say symlinks are never followed. Fix the wording.
6. relay-and-close -- mark CA1-CA5 resolved/documented, refresh CHANGES + wiki,
   mark the saga shipped, rebuild binaries. --done.
