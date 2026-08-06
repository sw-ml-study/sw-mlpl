# Saga: mlplunit-round-2
Their section 8: (1) in-language-event-reporting -- a stateful
MLPL reporter sink needs persistent state across calls;
explicit global_set(name, value) escape hatch (binding hygiene
stays default; writes recorded outside the frame snapshot and
replayed outward). (2) language-native-runner -- sandboxed fs
API (fs_walk/read_text/write_text/remove_path) + run_script
(fresh env, include-preserving, structured status + captured
typed events). Assessment: docs/q-and-a.md 2026-08-07.
## Steps
1. global-set -- the escape hatch + frame replay; their
   reporter fixture pattern as TDD.
2. fs-api -- sandboxed fs builtins over the FsProvider seam.
3. run-script -- fresh-env script execution with structured
   result + captured events.
4. close -- docs, gate run expecting both AVAILABLE, q-and-a.
