//! Demo-dropdown grouping + device-aware gating, and the connect-time
//! `/v1/devices` probe hook. Split from `mode_bar` so each module stays
//! within the function-count budget.

use yew::prelude::*;

use mlpl_web_demos::{DEMOS, Device, capability_for, demo_disabled};

/// One dropdown option: `(demo index, name, disabled)`.
pub type DemoOption = (usize, &'static str, bool);
/// Dropdown sections: `(section label, options)`.
pub type DemoGroups = Vec<(&'static str, Vec<DemoOption>)>;

/// Curated dropdown section order (user direction: logical, not
/// alphabetical, at the GROUP level -- this is an ML-oriented
/// language, so the ML learning path leads and the APL2/general-
/// programming group follows; gated connect tiers close the list).
/// Within each section, demo names still sort alphabetically.
/// Unlisted future sections land between the known CPU groups and
/// the connect tiers, alphabetically.
const SECTION_ORDER: &[&str] = &[
    "Basics",
    "Training & Learning",
    "Classical ML",
    "Classification",
    "Clustering",
    "Dim Reduction",
    "Vision",
    "Attention",
    "Sequence Models",
    "Language Models",
    "Generative Models",
    "APL2 / General Programming",
    "Client-server (connect)",
    "MLX - Apple GPU (connect)",
    "CUDA - Linux GPU (connect)",
];

/// Group demos for the dropdown by capability tier. `cpu`/live demos
/// keep their authored category; connect-only demos get their own
/// device-tier sections (MLX and CUDA deliberately separate).
///
/// `disabled` keys off the peer's REAL capability: a connect demo is
/// runnable only when `connected` AND the peer offers the demo's device
/// (every peer has cpu; an `mlx`/`cuda` demo needs that GPU in the
/// peer's `/v1/devices` set). So a CUDA demo lights up against a CUDA
/// peer but stays disabled against an MLX-only peer (and vice versa).
#[must_use]
pub fn grouped_demos(connected: bool, peer_devices: &[Device]) -> DemoGroups {
    let mut map: std::collections::BTreeMap<&str, Vec<DemoOption>> =
        std::collections::BTreeMap::new();
    for (i, d) in DEMOS.iter().enumerate() {
        let cap = capability_for(d.name);
        let section = match cap.device {
            Device::Mlx => "MLX - Apple GPU (connect)",
            Device::Cuda => "CUDA - Linux GPU (connect)",
            Device::Cpu if cap.requires_connect => "Client-server (connect)",
            Device::Cpu => d.category,
        };
        let disabled = demo_disabled(&cap, connected, peer_devices);
        map.entry(section).or_default().push((i, d.name, disabled));
    }
    for items in map.values_mut() {
        items.sort_by_key(|(_, name, _)| name.to_ascii_lowercase());
    }
    let mut groups: DemoGroups = map.into_iter().collect();
    groups.sort_by_key(|(section, _)| {
        // Unknown future sections slot just before the connect tiers.
        let rank = SECTION_ORDER
            .iter()
            .position(|s| s == section)
            .unwrap_or(SECTION_ORDER.len() - 3);
        (rank, section.to_ascii_lowercase())
    });
    groups
}

/// Tooltip text for a DISABLED demo option, explaining how to enable it.
/// Keyed off the dropdown section label (which encodes the device tier),
/// since a disabled GPU demo just needs the right connected peer.
#[must_use]
pub fn disabled_hint(section: &str) -> &'static str {
    if section.starts_with("CUDA") {
        "Available when connected to a server on a machine with an NVIDIA GPU (CUDA) -- e.g. a Linux box. Run mlpl-serve --features cuda there and connect to it."
    } else if section.starts_with("MLX") {
        "Available when connected to a server on an Apple-Silicon Mac (MLX GPU). Run mlpl-serve there and connect to it."
    } else {
        "Available when connected to an mlpl-serve instance. Append ?connect=<server-url> to this page's URL to point at a running server."
    }
}

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

/// Probe the connected peer's device set once on mount (async
/// `GET /v1/devices`), returning the resolved set. Empty until the
/// probe completes (and on the public, unconnected build) -- so GPU
/// demos start disabled and the right ones light up once the peer's
/// capability is known.
#[hook]
pub fn use_peer_devices() -> Vec<Device> {
    let peer = use_state(Vec::<Device>::new);
    {
        let peer = peer.clone();
        use_effect_with((), move |()| {
            yew::platform::spawn_local(async move {
                let names = mlpl_web_eval::devices::fetch_devices().await;
                peer.set(devices_from_names(&names));
            });
            || ()
        });
    }
    (*peer).clone()
}

/// "Connected" only counts when the server is actually reachable:
/// an https demo page cannot reach an http connect server (mixed
/// content), so gate those connect demos as disconnected rather
/// than enabling a button that will only fail.
#[must_use]
pub fn reachable_connected() -> bool {
    mlpl_web_eval::eval_url::is_connected()
        && mlpl_web_eval::connect_guard::connect_blocked_reason().is_none()
}
