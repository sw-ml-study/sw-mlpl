//! Shared demo types for the web playground demo cluster: the `Demo` struct,
//! the `Capability` gating tier (with the pure `demo_disabled` gate), and the
//! `ProgressNote` heads-up type. The per-demo DATA (which demos are
//! connect/GPU-gated, which have literate walkthroughs, which lines need a
//! heads-up note) lives in `demos.toml` and is generated into the
//! `mlpl-web-demos` facade -- this crate holds only the types + the pure gate.

pub struct Demo {
    pub name: &'static str,
    pub category: &'static str,
    pub intro: &'static str,
    pub takeaway: &'static str,
    pub lines: &'static [&'static str],
}

/// Which compute backend a demo targets. `Cpu` demos run anywhere,
/// including the in-browser WASM interpreter on the public live
/// demo. `Mlx` (Apple GPU) and `Cuda` (NVIDIA/Linux GPU) are
/// SEPARATE, connect-only groups -- each needs a `mlpl-serve` with
/// the matching device peer and never runs on the public live demo.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Mlx,
    Cuda,
}

/// A demo's runtime-requirement tier. Demos absent from the generated
/// `DEMO_CAPABILITIES` table default to [`Capability::CPU_LIVE`] --
/// runnable everywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// True when the demo needs a connected `mlpl-serve`; such
    /// demos render visible-but-not-runnable on the public live
    /// demo (which has no server).
    pub requires_connect: bool,
    pub device: Device,
}

impl Capability {
    /// The default tier: CPU, runnable on the public live demo.
    pub const CPU_LIVE: Self = Self {
        requires_connect: false,
        device: Device::Cpu,
    };
}

/// Whether a demo with capability `cap` should be DISABLED
/// (visible-but-not-runnable) in the UI, given the page connection
/// state and the connected peer's device set (from `GET /v1/devices`).
///
/// Gating keys off the peer's REAL capability, not a static guess: a
/// connect demo is runnable only when `connected` AND the peer offers
/// the demo's device. Every peer has `cpu`, so a cpu connect demo
/// needs only a connection; an `mlx`/`cuda` demo needs that GPU in the
/// peer's set -- so a CUDA demo lights up against a CUDA peer but stays
/// disabled against an MLX-only peer (and vice versa). Non-connect
/// (live) demos are always runnable.
#[must_use]
pub fn demo_disabled(cap: &Capability, connected: bool, peer_devices: &[Device]) -> bool {
    if !cap.requires_connect {
        return false;
    }
    let peer_offers = cap.device == Device::Cpu || peer_devices.contains(&cap.device);
    !(connected && peer_offers)
}

/// A heads-up rendered before a single long-running demo line.
/// Browser WASM evaluates each line on the main thread, so a
/// 30-step train block (Tiny LM) blocks the event loop for
/// seconds. Without a note the user sees a previous line's
/// output, then a stalled tab, then the result. The note paints
/// before the line starts so the wait is intentional and
/// estimated, not mysterious. The per-demo note DATA lives in
/// `demos.toml` (`[[progress_notes]]`), generated into the facade.
#[derive(Clone, Copy)]
pub struct ProgressNote {
    /// Demo's `name` field.
    pub demo: &'static str,
    /// Index into `Demo::lines` that the note precedes.
    pub line_idx: usize,
    /// Short heading -- e.g. "Training the language model".
    pub heading: &'static str,
    /// One-to-three-sentence body explaining what the runtime
    /// is about to do and a rough ETA on a recent laptop.
    pub body: &'static str,
}
