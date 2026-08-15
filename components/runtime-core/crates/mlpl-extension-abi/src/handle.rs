//! Opaque native-handle identity crossed at the extension
//! boundary: a provider-issued resource reference that MLPL passes
//! around by value without inspecting. The host never mints one
//! from MLPL numbers (non-forgeable) -- only a provider return
//! produces it, and the provider validates it on the way back.

/// A provider-issued handle: which extension minted it
/// (`extension_id`), the resource kind (`type_id`), and the
/// provider's slot-table coordinates (`slot` + `generation`). The
/// host treats all four as opaque bits and never interprets them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtHandle {
    pub extension_id: u64,
    pub type_id: u64,
    pub slot: u32,
    pub generation: u32,
}
