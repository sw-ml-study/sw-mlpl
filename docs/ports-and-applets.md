# Ports and applets

A **port** is an opaque handle connecting an MLPL program to an outside
service -- a native window, a background worker, any producer/consumer
of events and commands. It carries two channels: an event channel (the
service -> MLPL) and a command channel (MLPL -> the service). Only whole
values cross, so the two sides never share mutable memory.

Ports are minted by the service, not by MLPL, so a program cannot forge
one; it receives a port and passes it around by value.

## Port builtins

| Builtin | Meaning |
| --- | --- |
| `port_send(port, value)` | Send a command value to the service. |
| `port_recv(port)` | Block until one event arrives; returns it. |
| `port_poll(port)` | Non-blocking: drain all queued events. |
| `port_poll(port, limit)` | Non-blocking: drain at most `limit` events. |

`port_poll` returns a batch record `{count: N, items: {...}}`: `count`
is how many events were taken and `items` holds them in arrival order
(keys `000000`, `000001`, ...). An empty queue yields
`{count: 0, items: {}}`. Passing a `limit` is BOUNDED delivery -- a
burst of events cannot starve a single turn; the rest stay queued for
the next poll.

Events are records with a string `kind` field (`"key"`, `"pointer"`,
`"resize"`, `"close"`, ...) plus kind-specific fields.

## The applet model (register handlers, run a loop)

Rather than polling, a program can register handlers and hand control
to a dispatch loop -- the event-driven applet style:

| Builtin | Meaning |
| --- | --- |
| `on(port, "kind", :u:handler)` | Register a handler for an event kind. |
| `off(port, "kind")` | Unregister it. |
| `run(port, state)` | Dispatch loop: fold events through handlers. |

`run(port, state)` pulls each event, calls the handler registered for
its `kind`, and threads application state as a value:
`state = handler(event, state)`. It stops on a `"close"` event and
returns the final state. State is a plain value (typically a record),
so there is no shared mutable cell to corrupt.

```mlpl
# An applet: fold a counter, submit each new count as a command.
def u:on_tick(event, state) {
  next = state + 1
  port_send(port, next)
  next
}

on(port, "tick", :u:on_tick)
run(port, 0)          # returns the final count when a "close" arrives
```

A handler is a pure `(event, state) -> state` function; it may read the
event's fields, evolve the state, and submit commands with `port_send`.

## Native windows

A native window owns the operating-system main thread, so a windowed
applet runs the interpreter on a worker thread while the window's event
loop owns the main thread. The window (the service) forwards input as
event records and applies the commands the applet submits. This path is
available only when a program is launched as an applet on the local
main thread; over a remote connection a native-window request is a
clear error rather than a silent hang.

The service that drives a port -- a native window, or a test harness --
supplies the event stream and consumes the commands; the MLPL side is
the same whether the events come from a real window or a script.
