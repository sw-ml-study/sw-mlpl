# Using the MLX service

Saga R1 moved MLX from only an in-process Cargo feature to a
peer service shape. The normal topology is:

1. `mlpl-mlx-serve` runs on the Apple Silicon host.
2. `mlpl-serve` runs as the orchestrator.
3. Clients talk to the orchestrator with `mlpl-repl --connect`.

The orchestrator keeps CPU work local and forwards
`device("mlx") { ... }` blocks to the registered MLX peer.

## Start the peer

From the service workspace on the Apple Silicon host:

```sh
cd services/mlpl-mlx-serve
cargo run --release -- --bind 127.0.0.1:6465
```

Loopback bind is the default safe shape. For LAN use, bind the
peer to a reachable address and configure auth/firewalling at the
host boundary. The first R1 service protocol is intentionally small
and does not yet include peer discovery or a certificate trust model.

## Start the orchestrator

From the main workspace:

```sh
cargo run --release -p mlpl-serve -- --peer mlx=http://127.0.0.1:6465
```

For a non-loopback peer URL, pass `--insecure-peers` explicitly:

```sh
cargo run --release -p mlpl-serve -- \
  --peer mlx=http://192.168.1.10:6465 \
  --insecure-peers
```

The flag is deliberately loud because the R1 peer path is meant for
trusted development networks.

## Run a client

```sh
cargo run --release -p mlpl-repl -- --connect http://127.0.0.1:6464
```

Then run a remote MLX block:

```mlpl
x = device("mlx") {
  iota(5)
}

x_cpu = to_device("cpu", x)
```

The first expression returns an opaque device tensor handle in the
orchestrator session. `to_device("cpu", x)` fetches the bytes back
from the peer and rebinds the result as a normal CPU array.

## Behavior and limits

- `device("mlx") { ... }` forwards the whole block as one peer eval
  when the orchestrator has `--peer mlx=<url>`.
- If no peer is registered, the in-process MLX feature remains the
  fallback for single-host MLX builds.
- CPU operations on a peer tensor fault by default. Fetch explicitly
  with `to_device("cpu", tensor)` so network and device transfer costs
  stay visible.
- Peer sessions are created lazily and cached per orchestrator
  session. Dropping the orchestrator session drops the peer session
  handle association.
- R1 only materializes f64 tensors over the wire. Broader dtype
  coverage, peer discovery, peer-to-peer tensor migration, streaming,
  and cancellation are follow-up work.

See `demos/mlx_remote.mlpl` for a small end-to-end demo.
