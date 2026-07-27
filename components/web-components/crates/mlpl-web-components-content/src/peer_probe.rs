//! The connect-time `/v1/devices` peer-capability probe hook.
//! Split from `demo_gating` so each module stays within the
//! function-count budget.

use yew::prelude::*;

use mlpl_web_demos::Device;

/// Map `GET /v1/devices` name strings to [`Device`]s. Unknown dropped.
fn devices_from_names(names: &[String]) -> Vec<Device> {
    names
        .iter()
        .filter_map(|n| match n.as_str() {
            "cpu" => Some(Device::Cpu),
            "mlx" => Some(Device::Mlx),
            "cuda" => Some(Device::Cuda),
            _ => None,
        })
        .collect()
}

/// Probe the connected peer's device set on mount (async
/// `GET /v1/devices`, retried with bounded backoff -- a page loaded
/// while the server restarts must still light up GPU demos once it
/// answers). Empty until the probe lands (and on the public,
/// unconnected build), so GPU demos start disabled and the right
/// ones enable when the peer's capability is known.
#[hook]
pub fn use_peer_devices() -> Vec<Device> {
    let peer = use_state(Vec::<Device>::new);
    {
        let peer = peer.clone();
        use_effect_with((), move |()| {
            yew::platform::spawn_local(async move {
                let names = mlpl_web_eval::devices::fetch_devices_with_retry().await;
                peer.set(devices_from_names(&names));
            });
            || ()
        });
    }
    (*peer).clone()
}
