//! CPU utilization from Mach host tick counters -- the macOS analog of
//! Linux's `/proc/stat`. Unlike `top -l 2`, this lets the caller choose a
//! short sampling window and keeps live web telemetry responsive.

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

/// Busy CPU percentage between two `[user, system, idle, nice]` tick
/// snapshots. Returns `None` for an empty window or counter regression.
/// Counters are `natural_t` (u32), so `f64::from` is lossless.
#[must_use]
pub fn cpu_percent_between(before: [u32; 4], after: [u32; 4]) -> Option<f64> {
    let user = after[0].checked_sub(before[0])?;
    let system = after[1].checked_sub(before[1])?;
    let idle = after[2].checked_sub(before[2])?;
    let nice = after[3].checked_sub(before[3])?;
    let busy = user.checked_add(system)?.checked_add(nice)?;
    let total = busy.checked_add(idle)?;
    (total > 0).then(|| f64::from(busy) / f64::from(total) * 100.0)
}

/// Read the four aggregate Mach CPU tick counters
/// (`[user, system, idle, nice]`). `HOST_CPU_LOAD_INFO` is a
/// 32-bit `natural_t` flavor, so it is read with `host_statistics`
/// (NOT `host_statistics64`, which serves the 64-bit VM structs and
/// leaves these counters zeroed).
// `mach_host_self` is deprecated in `libc` in favour of the `mach2`
// crate, but `mach2` does not expose the host-statistics API we need,
// so the libc entry point stays -- it is a stable Mach trap.
#[allow(deprecated)]
#[must_use]
pub fn ticks() -> Option<[u32; 4]> {
    let mut info = libc::host_cpu_load_info { cpu_ticks: [0; 4] };
    let mut count = libc::HOST_CPU_LOAD_INFO_COUNT;
    // SAFETY: `info` is a correctly sized writable host_cpu_load_info and
    // `count` names that size in natural_t units, as required by Mach.
    let status = unsafe {
        libc::host_statistics(
            libc::mach_host_self(),
            libc::HOST_CPU_LOAD_INFO,
            (&raw mut info).cast::<libc::integer_t>(),
            &raw mut count,
        )
    };
    (status == libc::KERN_SUCCESS).then_some(info.cpu_ticks)
}

/// Busy CPU% from `top -l 2 -n 0` (two samples; the second is the live
/// interval). Blocks ~1s for top's sampling window, so callers run it off
/// the async runtime. `None` off-macOS or on any spawn/parse failure.
#[must_use]
pub fn percent() -> Option<f64> {
    let before = ticks()?;
    std::thread::sleep(std::time::Duration::from_millis(120));
    cpu_percent_between(before, ticks()?)
}
