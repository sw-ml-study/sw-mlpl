//! Unit tests for the pure `parse_*` cores of each Linux source. The
//! IO wrappers (`sample`/`usage`/`query`) are environment-dependent and
//! covered by the live `/v1/stats` smoke test in mlpl-serve.

use mlpl_monitor_linux::{cpu, gpu, mem};

#[test]
fn parse_stat_sums_total_and_idle() {
    // cpu  user nice system idle iowait irq softirq ...
    let text = "cpu  100 0 50 800 40 0 10 0 0 0\ncpu0 25 0 12 200 10 0 2\n";
    let (total, idle) = cpu::parse_stat(text).expect("aggregate cpu line parses");
    assert_eq!(total, 100 + 50 + 800 + 40 + 10);
    assert_eq!(idle, 800 + 40); // idle + iowait
}

#[test]
fn parse_stat_rejects_missing_cpu_line() {
    assert_eq!(cpu::parse_stat("intr 1 2 3\nctxt 99\n"), None);
}

#[test]
fn parse_meminfo_computes_used_in_mb() {
    let text = "MemTotal:       32000000 kB\nMemFree:  1000000 kB\nMemAvailable:    8000000 kB\n";
    let (used_mb, total_mb) = mem::parse_meminfo(text).expect("meminfo parses");
    assert_eq!(total_mb, 32_000_000 / 1024);
    assert_eq!(used_mb, (32_000_000 - 8_000_000) / 1024);
}

#[test]
fn parse_meminfo_rejects_missing_available() {
    assert_eq!(mem::parse_meminfo("MemTotal: 32000000 kB\n"), None);
}

#[test]
fn parse_smi_reads_first_gpu_row() {
    let (pct, used, total) = gpu::parse_smi("0, 278, 16311\n1, 9000, 16311\n").expect("smi parses");
    assert!((pct - 0.0).abs() < f64::EPSILON);
    assert_eq!(used, 278);
    assert_eq!(total, 16311);
}

#[test]
fn parse_smi_rejects_short_row() {
    assert_eq!(gpu::parse_smi("88\n"), None);
}
