//! Unit tests for the pure macOS parsers (`vm_stat` + `ioreg`), exercised
//! on fixed fixtures so they need no live host.

use mlpl_monitor_macos::cpu::{cpu_percent_between, parse_cpu_percent};
use mlpl_monitor_macos::gpu::parse_ioreg;
use mlpl_monitor_macos::mem::parse_used_bytes;

#[test]
fn top_cpu_percent_uses_last_sample_and_is_100_minus_idle() {
    // top -l 2 prints two "CPU usage" lines; the SECOND is the live
    // interval. busy = 100 - 86.67 = 13.33.
    let out = "Processes: 500 total\n\
CPU usage: 2.00% user, 3.00% sys, 95.00% idle\n\
... second sample ...\n\
CPU usage: 5.33% user, 8.00% sys, 86.67% idle\n";
    let pct = parse_cpu_percent(out).expect("parses");
    assert!((pct - 13.33).abs() < 0.01, "pct was {pct}");
}

#[test]
fn top_without_a_cpu_line_is_none() {
    assert_eq!(parse_cpu_percent("Processes: 500 total\n"), None);
}

#[test]
fn mach_tick_delta_reports_busy_percent() {
    // user + system + nice advance by 60 ticks; idle advances by 40.
    let before = [100, 200, 300, 10];
    let after = [130, 220, 340, 20];
    assert_eq!(cpu_percent_between(before, after), Some(60.0));
}

#[test]
fn mach_tick_delta_rejects_empty_or_wrapped_windows() {
    assert_eq!(cpu_percent_between([1; 4], [1; 4]), None);
    assert_eq!(cpu_percent_between([2; 4], [1; 4]), None);
}

// Live host check (macOS only): the Mach tick reader must return real,
// advancing counters -- the regression read them via host_statistics64,
// which left them zeroed so `percent()` was always None (flat
// sparklines). Retries absorb the timing jitter of a short window (Mach
// may hand back an identical cached snapshot); the regression makes
// EVERY attempt fail, so the retry loop still catches it.
#[cfg(target_os = "macos")]
#[test]
fn mach_ticks_are_live_and_percent_is_some() {
    use mlpl_monitor_macos::cpu::{percent, ticks};
    let advances = (0..5).any(|_| {
        let a = ticks().expect("host_statistics HOST_CPU_LOAD_INFO should succeed on macOS");
        std::thread::sleep(std::time::Duration::from_millis(150));
        ticks().expect("second ticks read") != a
    });
    assert!(advances, "CPU tick counters never advanced across reads");
    let sampled = (0..5).find_map(|_| percent());
    let p = sampled.expect("percent() must return a live sample within a few tries");
    assert!((0.0..=100.0).contains(&p), "cpu percent out of range: {p}");
}

const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                               35457.\n\
Pages active:                            452290.\n\
Pages inactive:                          305123.\n\
Pages speculative:                       148248.\n\
Pages wired down:                        100000.\n\
Pages occupied by compressor:             50000.\n";

#[test]
fn vm_stat_used_is_active_plus_wired_plus_compressed_times_page() {
    // (452290 + 100000 + 50000) pages * 16384 bytes.
    let want = (452_290u64 + 100_000 + 50_000) * 16_384;
    assert_eq!(parse_used_bytes(VM_STAT), Some(want));
}

#[test]
fn vm_stat_missing_field_is_none() {
    let truncated = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                               35457.\n";
    assert_eq!(parse_used_bytes(truncated), None);
}

const IOREG: &str = "    \"PerformanceStatistics\" = {\"In use system memory (driver)\"=0,\
\"Tiler Utilization %\"=5,\"Device Utilization %\"=5,\"In use system memory\"=637435904}\n\
    \"model\" = \"Apple M3 Pro\"\n";

#[test]
fn ioreg_reads_util_used_name_and_skips_driver_memory() {
    let total = 19_327_352_832u64; // 18 GiB
    let (name, pct, used_mb, total_mb) = parse_ioreg(IOREG, total).expect("parses");
    assert_eq!(name, "Apple M3 Pro");
    assert!((pct - 5.0).abs() < f64::EPSILON, "pct was {pct}");
    // 637435904 bytes -> 607 MiB; not the (driver)=0 value.
    assert_eq!(used_mb, 637_435_904 / (1024 * 1024));
    assert_eq!(total_mb, total / (1024 * 1024));
}

#[test]
fn ioreg_without_utilization_is_none() {
    assert_eq!(parse_ioreg("\"model\" = \"Apple M3 Pro\"", 1024), None);
}
