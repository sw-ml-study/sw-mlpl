//! Demo-dropdown grouping + device-aware gating. Split from
//! `mode_bar` (and the probe hook out to `peer_probe`) so each
//! module stays within the function-count budget.

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
            Device::Ollama => "Client-server (connect)",
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
    groups.sort_by_key(|(section, _)| section_rank(section));
    groups
}

/// Sort key for a dropdown section: its curated [`SECTION_ORDER`]
/// position. Unknown future sections slot just before the connect
/// tiers, alphabetically.
fn section_rank(section: &str) -> (usize, String) {
    let rank = SECTION_ORDER
        .iter()
        .position(|s| *s == section)
        .unwrap_or(SECTION_ORDER.len() - 3);
    (rank, section.to_ascii_lowercase())
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
        "Available when connected to a REACHABLE mlpl-serve instance (the Ask Ollama demo also needs Ollama running on that server's host). Append ?connect=<server-url> to this page's URL -- usually the page's own origin, e.g. ?connect=http://large12:6464 when the page is served from large12:6464."
    }
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
