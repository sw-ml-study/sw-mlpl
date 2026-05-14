//! Saga 21.5 step 005: viz-format detection table.
//!
//! `VizFormat::detect` recognizes SVG, HTML, PNG, JPEG (by magic
//! bytes / text-prefix sniffing) plus an explicit
//! `application/json` opt-in. Returns `None` when no format
//! matches; callers fall back to "store opaquely" (the
//! `/v1/viz` POST handler) or "decline to cache" (the local
//! `MLPL_CACHE_DIR` write path).
//!
//! Each variant carries its `content_type` (the MIME for
//! `/v1/viz` responses) and `extension` (used by
//! `viz_cache::write_bytes_to_cache` to pick the right
//! `<hash>.<ext>` suffix).

/// Kinds of viz payload the cache + storage layer can identify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VizFormat {
    Svg,
    Html,
    Png,
    Jpeg,
    Json,
}

const PNG_MAGIC: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
const JPEG_MAGIC: &[u8] = &[0xFF, 0xD8, 0xFF];
const JSON_OPT_IN: &[u8] = b"#mlpl-viz: json";

impl VizFormat {
    /// Sniff the format from the raw bytes. JSON is opt-in
    /// (`#mlpl-viz: json` prefix) because eval routinely
    /// produces ordinary string values that happen to start
    /// with `{` -- treating those as viz would write noise to
    /// the cache dir.
    #[must_use]
    pub fn detect(bytes: &[u8]) -> Option<VizFormat> {
        if bytes.starts_with(PNG_MAGIC) {
            return Some(VizFormat::Png);
        }
        if bytes.len() >= 3 && &bytes[..3] == JPEG_MAGIC {
            return Some(VizFormat::Jpeg);
        }
        if bytes.starts_with(JSON_OPT_IN) {
            return Some(VizFormat::Json);
        }
        let text = std::str::from_utf8(bytes).ok()?;
        let trimmed = text.trim_start();
        if let Some(rest) = trimmed.strip_prefix("<?xml")
            && rest.trim_start().contains("<svg")
        {
            return Some(VizFormat::Svg);
        }
        if trimmed.starts_with("<svg") {
            return Some(VizFormat::Svg);
        }
        let lower_prefix: String = trimmed
            .chars()
            .take(15)
            .flat_map(char::to_lowercase)
            .collect();
        if lower_prefix.starts_with("<!doctype html") || lower_prefix.starts_with("<html") {
            return Some(VizFormat::Html);
        }
        None
    }

    /// MIME type for `Content-Type` headers on `/v1/viz`
    /// responses.
    #[must_use]
    pub fn content_type(self) -> &'static str {
        match self {
            VizFormat::Svg => "image/svg+xml",
            VizFormat::Html => "text/html",
            VizFormat::Png => "image/png",
            VizFormat::Jpeg => "image/jpeg",
            VizFormat::Json => "application/json",
        }
    }

    /// Filename extension (no leading dot) for cache-dir
    /// writes. `Jpeg` writes as `.jpg`.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            VizFormat::Svg => "svg",
            VizFormat::Html => "html",
            VizFormat::Png => "png",
            VizFormat::Jpeg => "jpg",
            VizFormat::Json => "json",
        }
    }
}
