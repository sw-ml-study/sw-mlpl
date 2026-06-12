//! System RAM from `vm_stat` (used pages) + `sysctl hw.memsize` (total).
//! Both read without sudo.

use std::process::Command;

/// Parse `vm_stat` output into used bytes = (active + wired + compressed)
/// pages * page size -- the "in use" memory Activity Monitor shows,
/// excluding free/inactive/speculative. The page size comes from the
/// header line `(page size of N bytes)`. `None` if the header or any of
/// the three page counts is missing.
#[must_use]
pub fn parse_used_bytes(text: &str) -> Option<u64> {
    let page: u64 = text
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |key: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(key))?
            .split(':')
            .nth(1)?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    };
    let used = pages("Pages active:")?
        + pages("Pages wired down:")?
        + pages("Pages occupied by compressor:")?;
    Some(used * page)
}

/// Read `(used_mb, total_mb)`: used from `vm_stat`, total from `sysctl
/// hw.memsize`. `None` off-macOS or on any spawn/parse failure.
#[must_use]
pub fn usage() -> Option<(u64, u64)> {
    let total: u64 = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())?
        .trim()
        .parse()
        .ok()?;
    let vm = Command::new("vm_stat").output().ok()?;
    let used = parse_used_bytes(&String::from_utf8(vm.stdout).ok()?)?;
    Some((used / (1024 * 1024), total / (1024 * 1024)))
}
