//! Demo-dropdown grouping + device-aware gating. Split from
//! `mode_bar` (and the probe hook out to `peer_probe`) so each
//! module stays within the function-count budget.

use std::collections::BTreeMap;

use mlpl_web_demos::{DEMOS, Device, capability_for, demo_disabled};

/// One dropdown option: `(demo index, name, disabled)`.
pub type DemoOption = (usize, &'static str, bool);
/// Dropdown sections: `(section label, options)`.
pub type DemoGroups = Vec<(&'static str, Vec<DemoOption>)>;

/// Category explanations for the custom picker group labels.
pub const GROUP_TOOLTIPS: &[(&str, &str)] = &[
    (
        "Basics",
        "Core MLPL syntax, arrays, functions, and first programs.",
    ),
    (
        "Training & Learning",
        "How models learn: objectives, gradients, optimization, and adaptation.",
    ),
    (
        "Experiment Quality",
        "Ways to make experiments inspectable, comparable, robust, and leakage-safe.",
    ),
    (
        "Classical ML",
        "Established statistical learning methods for tabular and numeric data.",
    ),
    (
        "Classification",
        "Predict discrete classes and inspect how classification decisions behave.",
    ),
    (
        "Clustering",
        "Discover groups and structure in unlabeled data.",
    ),
    (
        "Data Forge",
        "Create, reshape, label, and inspect datasets for repeatable experiments.",
    ),
    (
        "Dim Reduction",
        "Project high-dimensional data into smaller, interpretable representations.",
    ),
    (
        "Vision",
        "Image representation, convolution, recognition, and visual inspection.",
    ),
    (
        "Attention",
        "Attention mechanisms and the dataflow behind transformer-style models.",
    ),
    (
        "Sequence Models",
        "Models for ordered observations, recurrence, and temporal context.",
    ),
    (
        "Language Models",
        "Token models, generation, adaptation, and language-model internals.",
    ),
    (
        "Generative Models",
        "Models that learn a distribution and generate new samples.",
    ),
    (
        "Engram",
        "Explicit learned memory and retrieval inside neural models.",
    ),
    (
        "APL2 / General Programming",
        "Array-language techniques useful beyond machine learning.",
    ),
    (
        "Non-Browser (companion CLIs)",
        "Examples that require a native companion command instead of browser execution.",
    ),
    (
        "Client-server (connect)",
        "Examples that run through a connected mlpl-serve backend.",
    ),
    (
        "MLX - Apple GPU (connect)",
        "GPU examples requiring a connected Apple-Silicon MLX backend.",
    ),
    (
        "CUDA - Linux GPU (connect)",
        "GPU examples requiring a connected NVIDIA CUDA backend.",
    ),
];

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
    "Experiment Quality",
    "Classical ML",
    "Classification",
    "Clustering",
    "Dim Reduction",
    "Vision",
    "Attention",
    "Sequence Models",
    "Language Models",
    "Generative Models",
    "Engram",
    "APL2 / General Programming",
    "Non-Browser (companion CLIs)",
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
    let mut map = BTreeMap::<&str, Vec<DemoOption>>::new();
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
    map.values_mut()
        .for_each(|items| items.sort_by_key(|(_, name, _)| name.to_ascii_lowercase()));
    let mut groups: DemoGroups = map.into_iter().collect();
    groups.sort_by_key(|(section, _)| {
        let rank = SECTION_ORDER
            .iter()
            .position(|s| *s == *section)
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
        "Available when connected to a REACHABLE mlpl-serve instance (the Ask Ollama demo also needs Ollama running on that server's host). Append ?connect=<server-url> to this page's URL -- usually the page's own origin, e.g. ?connect=http://large12:6464 when the page is served from large12:6464."
    }
}

/// Demo summary plus an enablement explanation when capability gating disables it.
#[must_use]
pub fn demo_tooltip(index: usize, section: &str, disabled: bool) -> String {
    let demo = &DEMOS[index];
    let summary = demo
        .intro
        .split_once(". ")
        .map_or(demo.intro, |(first, _)| first);
    if disabled {
        format!(
            "{}: {summary}. Unavailable: {}",
            demo.name,
            disabled_hint(section)
        )
    } else {
        format!("{}: {summary}.", demo.name)
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
