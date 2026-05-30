# Split mlpl-build main.rs tests into a sibling module (saga 75)

main.rs has 9 production fns + 6 inline tests = 15 (FAIL).
Move tests to a separate module (build_tests.rs or similar)
to retire the Module Function Count FAIL.
