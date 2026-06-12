//! Host identity from `sysctl kern.hostname` (no sudo).

use std::process::Command;

/// The host's network name, or `None` off-macOS / on spawn failure /
/// when empty. Used so `:status` can show a friendly hostname before the
/// connect IP.
#[must_use]
pub fn hostname() -> Option<String> {
    Command::new("sysctl")
        .args(["-n", "kern.hostname"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
