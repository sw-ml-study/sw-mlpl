//! The connect-time `/v1/devices` peer-capability probe hook.
//! Split from `demo_gating` so each module stays within the
//! function-count budget.

use yew::prelude::*;

use mlpl_web_demos::Device;

/// Map `GET /v1/devices` name strings to [`Device`]s (including the
/// synthetic `"ollama"` liveness entry). Unknown names dropped.
fn devices_from_names(names: &[String]) -> Vec<Device> {
    names
        .iter()
        .filter_map(|n| match n.as_str() {
            "cpu" => Some(Device::Cpu),
            "mlx" => Some(Device::Mlx),
            "cuda" => Some(Device::Cuda),
            "ollama" => Some(Device::Ollama),
            _ => None,
        })
        .collect()
}

/// Outcome of the peer-capability probe: `done` distinguishes "still
/// retrying" from "gave up / unreachable" so the UI can show a
/// server-not-responding banner instead of silently-dark demos.
#[derive(Clone, PartialEq, Default)]
pub struct PeerProbe {
    pub devices: Vec<Device>,
    pub done: bool,
}

/// Probe the connected peer's device set on mount (async
/// `GET /v1/devices`, retried with bounded backoff -- a page loaded
/// while the server restarts must still light up GPU demos once it
/// answers). Devices stay empty until the probe lands (and on the
/// public, unconnected build), so GPU demos start disabled and the
/// right ones enable when the peer's capability is known.
#[hook]
pub fn use_peer_probe() -> PeerProbe {
    let peer = use_state(PeerProbe::default);
    {
        let peer = peer.clone();
        use_effect_with((), move |()| {
            yew::platform::spawn_local(async move {
                let names = mlpl_web_eval::devices::fetch_devices_with_retry().await;
                peer.set(PeerProbe {
                    devices: devices_from_names(&names),
                    done: true,
                });
            });
            || ()
        });
    }
    (*peer).clone()
}

/// The always-visible connect banner: an invalid/blocked `?connect=`
/// (validation or mixed-content) or an exhausted probe against an
/// unresponsive server. Empty Html when connect mode is healthy or
/// absent.
pub fn connect_banner(probe: &PeerProbe) -> Html {
    if let Some(reason) = mlpl_web_eval::connect_guard::connect_blocked_reason() {
        return banner_div(reason);
    }
    if mlpl_web_eval::eval_url::is_connected() && probe.done && probe.devices.is_empty() {
        let url = mlpl_web_eval::eval_url::current_connect_url_from_window().unwrap_or_default();
        return banner_div(format!(
            "Server at {url} is not responding (kept retrying for ~30s). Check that \
             mlpl-serve is running there, and that the ?connect= URL is right -- \
             usually the same host:port that serves this page."
        ));
    }
    html! {}
}

/// Shared banner markup: high-contrast, above the mode bar, no CSS
/// dependency so it renders identically on every deployment.
fn banner_div(msg: String) -> Html {
    html! {
        <div
            class="connect-banner"
            style="background:#7f1d1d;color:#fecaca;padding:6px 12px;\
                   border-radius:6px;margin:4px 0;font-size:0.9em;">
            { msg }
        </div>
    }
}
