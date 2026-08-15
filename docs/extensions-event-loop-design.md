# Extensions event loop -- the browser model (B6 design)

The interactive native-3D demo needs a live event loop on top of the
extension data boundary (dense arrays both directions, opaque native
handles, structured record returns -- all shipped in
`extension-arrays-handles`). This design gives MLPL a responsive
native window without freezing the UI and without data races, using
the same shape a web browser uses: a single-threaded event loop with
registered callbacks (the JS side) talking to a responsive UI thread
(the compositor) by message passing.

## Thread model

Two threads, sharing NOTHING mutable -- only owned messages over
channels:

```
  MAIN THREAD (Rust)                         MLPL WORKER THREAD
  UI host: winit + wgpu          events      sw-mlpl dispatch loop
  - owns all window/GPU     ---------------> (its OWN single-threaded
    state (provider)           (owned msgs)    event loop) pulls an event,
  - renders continuously                       CALLS the registered handler
    -> always responsive                       u:on_key(...)  <- callback,
  - forwards input/resize/    commands           in-process, on THIS thread
    close as messages       <---------------   handler submits a command
  - drains render commands    (owned msgs)     + evolves app state (a value)
```

The load-bearing fact: **the callback into sw-mlpl fires on the MLPL
worker thread, not the UI thread.** The UI thread never touches the
interpreter -- it only forwards an event message. sw-mlpl's own event
loop (on the worker) receives it and invokes the MLPL handler
in-process. That is what makes three things true at once:

- **Responsive UI** -- the render thread is independent; a slow handler
  delays only MLPL's own callbacks (exactly like JS), while the
  compositor keeps painting the last scene and handling drag / resize /
  close.
- **No data races** -- the two threads share only owned messages;
  Rust's `Send` bound proves the cross-thread half, and the MLPL side
  is single-threaded so handlers never overlap (the intra-thread half).
  No locks, no `Arc<Mutex>`.
- **JS-applet ergonomics** -- MLPL registers handlers and the system
  calls them back.

## The MLPL primitive: `Port`

A general, reusable concurrency-with-the-outside-world primitive, not a
native3d hack. A `Port` is a native handle backed by a command channel
(MLPL -> far end) and an event channel (far end -> MLPL):

```
p = native3d:open({title: "demo", w: 800, h: 600})   # returns a Port

native3d:on(p, "key",   :u:on_key)     # addEventListener
native3d:on(p, "frame", :u:on_frame)
native3d:on(p, "close", :u:on_close)

native3d:run(p)      # hand control to the dispatch loop (app.mainloop /
                     # the browser never returning). Runs on the worker;
                     # the UI thread is unaffected and stays responsive.
```

Supporting builtins are general over any `Port`:

- `port_send(p, value)` -- push a command to the far end (send).
- `port_poll(p)` -- non-blocking drain of the event queue (try_recv),
  returning a record batch; the pull alternative to `run`.
- `port_recv(p)` -- blocking receive of one event.
- `on(p, event, :u:fn)` / `off(p, event)` -- register / unregister a
  handler (stored host-side; safe to mutate mid-loop since dispatch is
  single-threaded on the worker).

The same primitive later fronts any async native service (a socket, a
sensor, a background worker), so it is a real sw-mlpl general-
programming story.

## State sharing

Two kinds, both race-free without locks:

- **Native resources** (window, GPU buffers, scene) live on the UI
  thread, mutated only there; MLPL affects them by *submitting
  commands*, never by touching them.
- **App/logic state** is threaded as an ordinary MLPL value (a Record):
  each handler is a pure `(event, state) -> state`, and the dispatch
  loop owns the single copy and folds events into it. No shared mutable
  reference, so nothing to corrupt. (We deliberately do NOT add a
  mutable `cell`/`ref` primitive -- value threading gives shared state
  without aliasing.)

## Division of labor

- **sw-mlpl** owns: the launch mode (interpreter on a worker), the
  `Port` primitive + channels, the handler registry + dispatch loop,
  and a main-thread "UI host" hook the provider plugs into. Ships with
  a HEADLESS host double so the whole path is CI-testable without a
  window.
- **The provider** (`demo-extensions`, wgpu/winit) owns: the UI host on
  the main thread, forwarding events into the port's event channel and
  draining commands from its command channel, via thread-safe C-ABI
  `poll(port, max) -> events` + `submit(port, cmd)` entries. No
  host-callback crosses the C ABI and no Rust channel crosses it -- the
  provider's UI thread and its C entries talk over the provider's own
  internal channels.

## Gate #1 (proven)

The whole design rests on the interpreter running on a spawned worker
while winit owns the main thread. PROVEN by `worker_thread_spike`:
`Environment` and `Value` are `Send`; an eval runs on a spawned thread;
an env built on the main thread MOVES into a worker and evals there.

## Non-goals (separate or deferred)

- Parallel MLPL / a thread-safe interpreter -- the MLPL side stays
  single-threaded (its safety model, like JS).
- A main-thread handoff for connect / serve -- native-UI ports are
  LOCAL-ONLY (the local main-thread launch path); connect / serve
  returns a clear error.
- The real wgpu renderer -- that is the provider; this design builds the
  sw-mlpl side plus a headless test double.
- Pub/sub topics / multiple subscribers per event -- one handler per
  event to start; a subscriber list is a later refinement.
- Dynamic loading (B3), compiler parity (B2), `use`-facade (B1).

## The hard parts the saga must land

1. **Launch inversion** -- process entry keeps the main thread free and
   runs the interpreter on a worker; the main thread parks until a
   native-UI port opens, then runs the UI host loop. (Gate #1 proven.)
2. **Main-thread event-loop timing** -- winit's `EventLoop` must be
   created on the main thread, so the runtime must know at startup that
   a native window may come (a launch mode / dedicated entry).
3. **Port + channels + dispatch loop** -- the general primitive above,
   with a headless host double for tests.
4. **Provider C-ABI contract** -- the thread-safe `poll` / `submit`
   entries + the main-thread / lifecycle rules the provider implements.
