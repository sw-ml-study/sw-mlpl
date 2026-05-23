Saga 33 step 004: extract env_device.rs + env_peer.rs from env.rs.

Continue env.rs split (now ~34 methods after step 003).

The device-stack + peer-dispatcher methods cluster naturally because peer dispatch is driven by device. But 12 methods in one impl block would FAIL fn-count -- split into two siblings:

Move to crates/mlpl-eval/src/env_device.rs (impl Environment block):
- device, push_device, pop_device, take_mlx_fallback_warning
- tensor_device, set_tensor_device
(6 methods -- PASS)

Move to crates/mlpl-eval/src/env_peer.rs (impl Environment block):
- set_peer_dispatcher, clear_peer_dispatcher, peer_dispatcher
- set_device_tensor, get_device_tensor, remove_device_tensor
(6 methods -- PASS)

Note: the PeerDispatcher trait + the device-tensor stash are conceptually related to remote MLX dispatch. Keep them in env_peer.rs.

Register both in lib.rs.

Target: env.rs 34 -> 22 methods. Approaching the budget.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1.