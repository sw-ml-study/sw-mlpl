//! Host identity from `/proc/sys/kernel/hostname`.

/// The host's network name, or `None` off-Linux / on read failure /
/// when empty. Used so `:status` can show a friendly hostname before
/// the connect IP.
#[must_use]
pub fn hostname() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
