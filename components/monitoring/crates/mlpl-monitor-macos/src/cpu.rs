//! CPU utilization by scraping `top` -- the no-sudo macOS path.
//!
//! The Mach `host_statistics64` tick counters would be the library
//! analog of Linux's `/proc/stat`, but libc's bindings for them are
//! deprecated and `mach2` (their suggested replacement) does not expose
//! the host-statistics API, so we parse `top` instead. `top -l 1` reports
//! CPU usage since boot (a near-constant average); `-l 2` takes a second
//! sample whose "CPU usage" line is the live interval -- that is the one
//! we read.

use std::process::Command;

/// Parse the LAST `CPU usage: U% user, S% sys, I% idle` line of `top -l 2`
/// output (the live interval sample) into busy percent = `100 - idle`.
/// `None` if no such line is present or the idle figure is unparseable.
#[must_use]
pub fn parse_cpu_percent(text: &str) -> Option<f64> {
    let line = text.lines().rfind(|l| l.contains("CPU usage:"))?;
    // " ... , 86.67% idle" -> the last comma segment carries idle.
    let idle: f64 = line
        .rsplit(',')
        .next()?
        .trim()
        .split('%')
        .next()?
        .trim()
        .parse()
        .ok()?;
    Some((100.0 - idle).clamp(0.0, 100.0))
}

/// Busy CPU% from `top -l 2 -n 0` (two samples; the second is the live
/// interval). Blocks ~1s for top's sampling window, so callers run it off
/// the async runtime. `None` off-macOS or on any spawn/parse failure.
#[must_use]
pub fn percent() -> Option<f64> {
    let out = Command::new("top")
        .args(["-l", "2", "-n", "0"])
        .output()
        .ok()?;
    parse_cpu_percent(&String::from_utf8(out.stdout).ok()?)
}
