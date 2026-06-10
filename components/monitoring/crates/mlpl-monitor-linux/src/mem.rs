//! System RAM from `/proc/meminfo`.

/// Parse `/proc/meminfo` into `(used_mb, total_mb)` where
/// `used = MemTotal - MemAvailable`. The file reports kibibytes; we
/// divide by 1024 to megabytes. `None` if either key is missing.
#[must_use]
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let kb = |key: &str| {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|n| n.parse::<u64>().ok())
    };
    let total = kb("MemTotal:")?;
    let avail = kb("MemAvailable:")?;
    Some((total.saturating_sub(avail) / 1024, total / 1024))
}

/// Read `/proc/meminfo` and return `(used_mb, total_mb)`. `None`
/// off-Linux or on any read/parse failure.
#[must_use]
pub fn usage() -> Option<(u64, u64)> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}
