//! Unit tests for the pure `:status` / sparkline rendering helpers.

use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::render::{gb, pct, status_lines};

#[test]
fn gb_converts_mb_to_gib_one_decimal() {
    assert_eq!(gb(Some(16311)), "15.9"); // 16311 / 1024
    assert_eq!(gb(Some(0)), "0.0");
}

#[test]
fn gb_absent_is_na() {
    assert_eq!(gb(None), "n/a");
}

#[test]
fn pct_rounds_to_whole_percent() {
    assert_eq!(pct(Some(12.925)), "13%");
    assert_eq!(pct(Some(0.0)), "0%");
    assert_eq!(pct(None), "n/a");
}

#[test]
fn status_lines_cover_all_four_resources() {
    let s = Snapshot {
        cpu_pct: Some(5.5),
        ram_used_mb: Some(9128),
        ram_total_mb: Some(257_291),
        gpu_pct: Some(0.0),
        vram_used_mb: Some(10),
        vram_total_mb: Some(16311),
    };
    let lines = status_lines(&s);
    assert_eq!(lines.len(), 4);
    assert!(lines[0].contains("CPU") && lines[0].contains("6%"));
    assert!(lines[1].contains("RAM") && lines[1].contains("251.3") && lines[1].contains("GB"));
    assert!(lines[2].contains("GPU") && lines[2].contains("0%"));
    assert!(lines[3].contains("VRAM") && lines[3].contains("15.9"));
}

#[test]
fn status_lines_render_na_for_absent_sources() {
    let lines = status_lines(&Snapshot::default());
    assert!(lines[0].contains("n/a")); // CPU
    assert!(lines[3].contains("n/a")); // VRAM
}
