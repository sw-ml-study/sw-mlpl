# Connect Topology -- how the playground finds its backends

Design record for the 2026-07-27 connect-UX overhaul (saga
eval-decomposition steps 002-006), plus the agreed direction for
multi-server setups. The guiding rule: **every "enabled" claim in
the UI must be backed by live evidence**, and the common one-server
case must need zero configuration.

## The default: same-origin auto-connect

`mlpl-serve --static-dir` serves the playground UI and the `/v1`
API from ONE listener, so the page's own origin is almost always
the right connect target. On load, when the URL has no `?connect=`
parameter at all, the app probes `<page-origin>/v1/devices`:

- It answers like an mlpl-serve -> the origin is adopted as the
  connect target (via `history.replaceState`, no reload) before the
  app mounts. `http://large12:6464/sw-mlpl/` just works.
- It does not answer (github.io, `trunk serve`, any static host) ->
  the page stays browser-local. No error: not connecting is the
  normal state for a static demo.

`?connect=off` is the explicit-disconnect sentinel: the Disconnect
button writes it (instead of clearing the parameter) so the reload
does not immediately auto-reconnect. The Connect button's panel
offers the serving origin back ("<host> (the server hosting this
page)") whenever its `/v1/devices` answers -- so
disconnect -> connect is also the recovery path when a backend
(Ollama, a GPU server) came up AFTER the page loaded.

## The override: `?connect=<url>`

Only needed when the UI is served from one place and the mlpl-serve
runs somewhere else (trunk dev page on :9957, a static host, a
laptop pointing at a lab box). Rules learned the hard way:

- The value is validated at load (`http(s)://host:port`, nothing
  else); a malformed value (stray `)`, missing port) shows a red
  banner immediately.
- `localhost` in the parameter resolves on the machine running the
  BROWSER, not the page's server. Use the server's real hostname.
- Cross-origin means CORS: the SERVER must be started with the
  page's origin in `--cors-allow` (comma-separated list; each entry
  matches the browser's `Origin` header exactly -- scheme + host +
  port, so the `localhost` and `large12` spellings of the same page
  are two entries).

## CORS cannot come from the URL

A page parameter can never grant CORS access -- the allow-list is
the SERVER's declaration of which page origins it trusts. If a page
could configure it, any malicious page could read any local
server's responses; the model only protects anything because the
server side owns it. Consequence: third-system topologies always
need a server-side flag (`--cors-allow` on the API server), never a
page-side switch.

## Multi-backend: federate on the server, not in the browser

The rejected alternative was per-backend page parameters
(`?cuda=host:port&mlx=host:port&ollama=host:port`). It multiplies
exactly the things that hurt: N CORS configurations (every backend
must allow the page origin), N auth relationships in the browser,
and a client that must merge capability claims. The existing
server-side federation already answers the use case:

- The browser connects to ONE mlpl-serve.
- That serve delegates `device("mlx")` / `device("cuda")` blocks to
  peers registered with `--peer mlx=http://mac:6465` (server-to-
  server HTTP: no browser CORS involved at all, auth is pairwise).
- `/v1/devices` reports the union the UI should gate on, and the
  connect panel shows the per-backend verdicts (CUDA / MLX / Ollama
  available or unavailable).

So "UI on a non-GPU host, CUDA on a Linux box, MLX on a Mac" is:
serve the UI anywhere, `?connect=` (or auto-connect) to the CUDA
box's mlpl-serve, and give THAT serve a `--peer mlx=...`.

## Ollama resolution

Each mlpl-serve has exactly one effective Ollama host:
`--ollama-host` flag, else `OLLAMA_HOST` env, else the built-in
default (see `resolve_ollama`). `/v1/devices` reports whether that
host is actually alive right now (800ms probe), and the Ask Ollama
demo gates on it. When a peer chain someday offers several Ollama
hosts, resolution stays a server concern: the CONNECTED serve's
configuration wins, and a future `--ollama-prefer <peer>` flag on
that serve -- not a page parameter -- would redirect it. Not
implemented; recorded here so the flag lands server-side when
needed.

## Evidence-based UI states (shipped)

| Signal | Source |
| --- | --- |
| Button "Connecting..." | devices probe in flight (bounded ~30s retry) |
| Button "Connected [check]" | probe answered |
| Button "Connect [warn]" | blocked, invalid, or probe exhausted |
| Red banner at load | invalid `?connect=` / mixed content / server not responding |
| Demo enablement | probe's device set only -- cpu tiers included; Ask Ollama needs the live `ollama` flag |
| Connect panel | live per-backend availability from `/v1/devices` |
