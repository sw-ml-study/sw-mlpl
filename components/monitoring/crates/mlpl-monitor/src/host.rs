//! Host identity, platform-selected like [`crate::snapshot`].

/// This host's network name, or `None` when unavailable (unsupported
/// platform or read failure).
#[cfg(target_os = "linux")]
#[must_use]
pub fn hostname() -> Option<String> {
    mlpl_monitor_linux::host::hostname()
}

#[cfg(target_os = "macos")]
#[must_use]
pub fn hostname() -> Option<String> {
    mlpl_monitor_macos::host::hostname()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[must_use]
pub fn hostname() -> Option<String> {
    None
}
