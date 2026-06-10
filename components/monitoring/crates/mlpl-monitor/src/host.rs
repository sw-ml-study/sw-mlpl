//! Host identity, platform-selected like [`crate::snapshot`].

/// This host's network name, or `None` when unavailable (off-Linux
/// until a macOS source lands, or on read failure).
#[cfg(target_os = "linux")]
#[must_use]
pub fn hostname() -> Option<String> {
    mlpl_monitor_linux::host::hostname()
}

#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn hostname() -> Option<String> {
    None
}
