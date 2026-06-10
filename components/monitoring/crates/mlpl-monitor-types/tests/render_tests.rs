//! Unit tests for the pure `:status` / sparkline rendering helpers.

use mlpl_monitor_types::render::{gb, gpu_line, pct, snapshot_lines};
use mlpl_monitor_types::{Gpu, Snapshot};

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
fn gpu_line_shows_name_index_util_and_vram() {
    let g = Gpu {
        name: Some("NVIDIA RTX 5080".to_string()),
        pct: Some(88.0),
        vram_used_mb: Some(2100),
        vram_total_mb: Some(16311),
    };
    let line = gpu_line(0, &g);
    assert!(line.contains("GPU 0"));
    assert!(line.contains("NVIDIA RTX 5080"));
    assert!(line.contains("88%"));
    assert!(line.contains("2.1") && line.contains("15.9"));
}

#[test]
fn gpu_line_unnamed_falls_back() {
    assert!(gpu_line(1, &Gpu::default()).contains("GPU 1: GPU"));
}

#[test]
fn snapshot_lines_one_gpu() {
    let s = Snapshot {
        cpu_pct: Some(5.5),
        ram_used_mb: Some(9128),
        ram_total_mb: Some(257_291),
        gpus: vec![Gpu {
            name: Some("NVIDIA RTX 5080".to_string()),
            pct: Some(0.0),
            vram_used_mb: Some(10),
            vram_total_mb: Some(16311),
        }],
    };
    let lines = snapshot_lines(&s);
    assert_eq!(lines.len(), 3); // CPU, RAM, 1 GPU
    assert!(lines[0].contains("CPU") && lines[0].contains("6%"));
    assert!(lines[1].contains("RAM") && lines[1].contains("251.3"));
    assert!(lines[2].contains("GPU 0") && lines[2].contains("15.9"));
}

#[test]
fn snapshot_lines_two_gpus() {
    let g = Gpu {
        name: Some("X".to_string()),
        pct: Some(1.0),
        vram_used_mb: Some(1),
        vram_total_mb: Some(2),
    };
    let s = Snapshot {
        gpus: vec![g.clone(), g],
        ..Snapshot::default()
    };
    let lines = snapshot_lines(&s);
    assert_eq!(lines.len(), 4); // CPU, RAM, GPU 0, GPU 1
    assert!(lines[2].contains("GPU 0"));
    assert!(lines[3].contains("GPU 1"));
}

#[test]
fn snapshot_lines_no_gpu_says_none() {
    let lines = snapshot_lines(&Snapshot::default());
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("n/a")); // CPU
    assert!(lines[2].contains("GPU") && lines[2].contains("none"));
}
