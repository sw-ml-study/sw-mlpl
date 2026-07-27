//! The `/v1/devices` probe retry policy: bounded backoff so a page
//! loaded while the server is restarting still lights up GPU demos
//! once the server answers, without polling forever.

use mlpl_web_eval_core::devices::retry_delay_ms;

#[test]
fn schedule_is_bounded_and_monotone() {
    let mut delays = Vec::new();
    let mut attempt = 0;
    while let Some(d) = retry_delay_ms(attempt) {
        delays.push(d);
        attempt += 1;
        assert!(attempt < 20, "schedule must terminate");
    }
    assert!(delays.len() >= 3, "need a few retries to cover a restart");
    assert!(delays.windows(2).all(|w| w[0] <= w[1]), "backoff grows");
}

#[test]
fn schedule_covers_a_server_restart_window() {
    let total: u32 = (0..).map_while(retry_delay_ms).sum();
    assert!(
        (15_000..=120_000).contains(&total),
        "total cover {total}ms should span a realistic restart (15s..2min)"
    );
}

#[test]
fn schedule_gives_up() {
    assert_eq!(retry_delay_ms(100), None, "must not retry forever");
}
