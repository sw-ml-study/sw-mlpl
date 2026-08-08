# Saga: demo-memory-clock
demo-memory agent's #1 (highest-value) upstream request: a
high-resolution monotonic clock -- blocks inserts/sec,
lookups/sec, latency percentiles, honest wall-clock. Native
runtime builtin (runtime-core is native-only, like llm_call/
load); serves the CLI + connect mode where demo-memory runs;
present-tense capability boundary in the browser. The other 4
requests (fixed-width ints+bitops, packed layouts, RNG
streams, gen-state telemetry) are recorded + queued.
## Steps
1. clock -- clock_ms() monotonic-elapsed-ms builtin (process
   epoch), catalog row, TDD; docs/demo-memory-upstream.md
   tracking all 5 requests; queue entries for #2-#4.
2. close -- rebuild repl/serve, docs pins, wiki, --done.
