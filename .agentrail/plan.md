# Saga: native-test-events
mlplunit contract: typed test events (their
docs/native-test-events.md; gate tests/capabilities/
events_case.mlpl). Assessment: docs/q-and-a.md 2026-08-06
evening. In-language callback API + host transport separate
from stdout/stderr. sw-MLPL validates the ENVELOPE only;
counting/TAP/duration/output-capture stay runner-side.
## Steps
1. events-core -- test_event_sink(:u:f) + emit_test_event(record)
   with loud envelope validation and callback delivery. TDD.
2. events-transport -- --test-events <path> JSONL in script mode
   (ordered, synchronous, exact text) + connect-mode SSE
   test_event type; docs (lang-reference, glossary+pin, usage,
   catalog); rebuild repl+serve+pages.
3. events-close -- run mlplunit check-capabilities expecting
   native-test-events AVAILABLE; q-and-a + wiki; --done.
