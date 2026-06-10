//! Unit tests for the pure sparkline renderer.

use mlpl_monitor_types::spark::{metric_percents, sparkline};
use mlpl_monitor_types::{Gpu, Snapshot};

#[test]
fn empty_is_empty() {
    assert_eq!(sparkline(&[], 100), "");
}

#[test]
fn one_glyph_per_sample() {
    assert_eq!(sparkline(&[0, 50, 100], 100).chars().count(), 3);
}

#[test]
fn extremes_map_to_lowest_and_highest_bar() {
    let s = sparkline(&[0, 100], 100);
    let chars: Vec<char> = s.chars().collect();
    assert_eq!(chars[0], '\u{2581}'); // lowest bar
    assert_eq!(chars[1], '\u{2588}'); // full bar
}

#[test]
fn over_max_clamps_to_full() {
    assert_eq!(sparkline(&[250], 100), "\u{2588}");
}

#[test]
fn zero_max_does_not_panic() {
    // max 0 treated as 1; any positive sample clamps to full.
    assert_eq!(sparkline(&[0, 5], 0).chars().count(), 2);
}

#[test]
fn metric_percents_derives_cpu_ram_gpu_vram() {
    let s = Snapshot {
        cpu_pct: Some(42.6),
        ram_used_mb: Some(64),
        ram_total_mb: Some(256),
        gpus: vec![Gpu {
            name: None,
            pct: Some(88.2),
            vram_used_mb: Some(4),
            vram_total_mb: Some(16),
        }],
    };
    // [cpu, ram, gpu, vram]
    assert_eq!(metric_percents(&s), [43, 25, 88, 25]);
}

#[test]
fn metric_percents_no_gpu_is_zero() {
    let p = metric_percents(&Snapshot::default());
    assert_eq!(p, [0, 0, 0, 0]);
}
