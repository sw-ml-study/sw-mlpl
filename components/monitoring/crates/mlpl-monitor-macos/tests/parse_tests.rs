//! Unit tests for the pure macOS parsers (`vm_stat` + `ioreg`), exercised
//! on fixed fixtures so they need no live host.

use mlpl_monitor_macos::cpu::parse_cpu_percent;
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
    let want = (452290u64 + 100000 + 50000) * 16384;
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
