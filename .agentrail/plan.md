# Saga: extensions-event-loop

Unblock the LIVE interactive native-3D demo (demo-extensions upstream
B6 / A8). The extension DATA boundary shipped in
`extension-arrays-handles` (dense f64 arrays both directions, opaque
native handles, structured record returns). This saga adds the
CONCURRENCY + event-loop story so an MLPL program drives a responsive
native window without freezing the UI and without data races.

Full architecture: `docs/extensions-event-loop-design.md`.

## Architecture (browser model) -- decided 2026-08-15

Two threads, share-nothing message passing (Rust `Send`-enforced, no
locks):

- MAIN thread = UI host (winit + wgpu, provider-owned): renders
  continuously, always responsive; forwards input/resize/close as
  messages; drains render commands.
- WORKER thread = MLPL interpreter + a single-threaded dispatch loop
  (the "event loop") that invokes registered MLPL handlers in-process.

The callback into sw-mlpl fires ON the worker thread, triggered by an
event message forwarded from the UI thread -- the UI thread never
touches the interpreter. That is what gives JS-applet ergonomics + a
responsive UI + no races simultaneously (JS main-thread loop + the
compositor + postMessage, mapped onto MLPL).

MLPL primitive: a general `Port` (a native handle) backed by a command
channel (MLPL -> far) + event channel (far -> MLPL), with
`on`/`off`/`run` (register + dispatch loop) and
`port_send`/`port_poll`/`port_recv`. Reusable for any async native
service, not native3d-specific. App state threads as a value (a
Record) folded by the loop; native resources live on the UI thread and
are mutated only by submitting commands.

## What exists

- extension-arrays-handles shipped (abi/cabi/registry, ext_marshal
  send/recv): arrays both directions, opaque handles, record returns.
- Gate #1 PROVEN (`worker_thread_spike`): `Environment` + `Value` are
  `Send`; eval runs on a spawned worker; an env built on main MOVES to
  a worker and evals there.
- MLPL loops: `repeat N` / `for x in expr` / `train N`. No while/break
  (not needed -- the dispatch loop is the loop).
- Local REPL/script eval is on the main thread; connect/serve eval is
  on worker threads (so native-UI ports are local-only).

## Non-goals (separate or deferred)

- Parallel MLPL / a thread-safe interpreter (MLPL side stays
  single-threaded -- its safety model, like JS).
- Main-thread handoff for connect/serve (native UI is local-only;
  connect returns a clear err).
- The real wgpu renderer (that is the demo-extensions provider); this
  saga builds the sw-mlpl side + a HEADLESS host double for CI.
- Pub/sub topics / multiple subscribers per event (one handler per
  event to start).
- Dynamic loading (B3), compiler parity (B2), use-facade (B1).

## Steps (TDD each)

1. **worker-thread-gate** -- land `worker_thread_spike` as a kept
   regression proving gate #1 (Environment/Value `Send` + eval on a
   spawned worker + env moved main->worker). Small; commits the spike.
2. **port-and-channels** -- a general `Port` value (native handle)
   backed by a command channel (MLPL->far) + event channel
   (far->MLPL), share-nothing owned messages. Host plumbing +
   `port_send` / `port_poll` (non-blocking, record batch) /
   `port_recv`. TDD with an in-process echo far-end (no winit).
3. **handler-registry-and-dispatch** -- `on(p, event, :u:fn)` /
   `off(p, event)` store fn-refs host-side; `run(p)` is the dispatch
   loop: pull events, invoke the matching MLPL handler in-process,
   thread app state as a folded value, stop on a close event. TDD
   headless: a scripted event sequence drives handlers; state evolves;
   close stops cleanly.
4. **ui-host-launch-inversion** -- the "parked main" UI host: process
   entry keeps main free and runs the interpreter on a worker; main
   parks until a native-UI port opens, then runs the UI host loop (a
   HEADLESS, winit-less double for CI). Policy: native-UI ports require
   the local main-thread launch path; connect/serve returns a clear
   err. TDD: an MLPL applet opens a port, the headless host forwards a
   scripted event stream + drains commands, handlers run on the worker,
   quit is clean.
5. **provider-abi-and-contract** -- the C-ABI shape the demo-extensions
   provider implements: thread-safe `poll(port, max) -> events` +
   `submit(port, cmd)` entries backed by the provider's internal
   channels; the main-thread / lifecycle / non-blocking-pump contract.
   A #[repr(C)] headless provider in tests exercises the full path.
6. **docs-demo-close** -- user-facing `Port` / `on` / `off` / `run` /
   `submit` docs (WHAT/HOW only), the demo-extensions upstream-contract
   note, wiki errata; a worked MLPL applet example (headless-runnable);
   queue follow-ons (responsive-worker escalation, pub/sub topics,
   connect-mode main-thread handoff). Rebuild. `--done`.
