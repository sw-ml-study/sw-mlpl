//! CPU utilization from `/proc/stat`.
//!
//! Utilization needs two samples over time, so this crate only exposes
//! a single `sample()` (and the pure `parse_stat`); the facade
//! (`mlpl-monitor`) takes two samples around a sleep and computes the
//! busy fraction.

/// Parse the aggregate `cpu ` line of `/proc/stat` into
/// `(total_jiffies, idle_jiffies)`. Idle = `idle + iowait` (fields 4
/// and 5). `None` if the line is absent or has too few fields.
#[must_use]
pub fn parse_stat(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().find(|l| l.starts_with("cpu "))?;
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    let total: u64 = nums.iter().sum();
    let idle = nums
        .get(3)?
        .saturating_add(nums.get(4).copied().unwrap_or(0));
    Some((total, idle))
}

/// Read `/proc/stat` and return one `(total, idle)` jiffy sample.
/// `None` off-Linux or on any read/parse failure.
#[must_use]
pub fn sample() -> Option<(u64, u64)> {
    parse_stat(&std::fs::read_to_string("/proc/stat").ok()?)
}
