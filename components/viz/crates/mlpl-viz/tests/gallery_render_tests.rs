//! `render_gallery` tests. Saga 29 step 010.

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{VizError, render, render_attention_overlay, render_gallery, render_with_aux};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn gallery_rejects_non_rank4_input() {
    let x = arr(vec![3, 64, 64], vec![0.0; 3 * 64 * 64]);
    let err = render_gallery(&x, None).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
}

#[test]
fn gallery_rejects_wrong_channel_count() {
    let x = arr(vec![2, 4, 8, 8], vec![0.0; 2 * 4 * 8 * 8]);
    let err = render_gallery(&x, None).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
}

#[test]
fn gallery_renders_single_image() {
    // 1x3x4x4 rainbow-ish gradient: each channel ramps.
    let mut data: Vec<f64> = Vec::with_capacity(3 * 4 * 4);
    for c in 0..3 {
        for y in 0..4 {
            for x in 0..4 {
                let v = match c {
                    0 => (x as f64) / 3.0,
                    1 => (y as f64) / 3.0,
                    _ => ((x + y) as f64) / 6.0,
                };
                data.push(v * 2.0 - 1.0); // [0, 1] -> [-1, 1]
            }
        }
    }
    let x = arr(vec![1, 3, 4, 4], data);
    let svg = render_gallery(&x, None).expect("render gallery");
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
    // At least one fill="rgb(...)" rect was emitted per pixel.
    let rect_count = svg.matches("<rect").count();
    assert!(
        rect_count > 16,
        "expected at least 16 pixel rects, got {rect_count}"
    );
}

#[test]
fn gallery_layout_uses_grid_for_multi_image_batch() {
    // 4 images -> 2x2 grid expected.
    let n = 4;
    let h = 4;
    let w = 4;
    let mut data: Vec<f64> = Vec::with_capacity(n * 3 * h * w);
    for i in 0..n {
        for c in 0..3 {
            for _ in 0..(h * w) {
                // Distinct intensity per image / channel so the
                // SVG output reflects all four.
                data.push((i as f64 + c as f64 * 0.1) / 6.0 * 2.0 - 1.0);
            }
        }
    }
    let x = arr(vec![n, 3, h, w], data);
    let svg = render_gallery(&x, None).expect("render gallery");
    // For a 2x2 thumbnail grid each image's top-left corner
    // should land at a distinct (x, y). We don't try to
    // pin exact pixel positions (those depend on padding /
    // canvas math); just check we got enough rects.
    let rect_count = svg.matches("<rect").count();
    assert!(
        rect_count >= n * 4,
        "expected >= {} rects across the grid, got {}",
        n * 4,
        rect_count
    );
}

#[test]
fn gallery_handles_empty_batch() {
    let x = arr(vec![0, 3, 4, 4], vec![]);
    let svg = render_gallery(&x, None).expect("render gallery on empty batch");
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn gallery_clamps_out_of_range_pixels_without_panicking() {
    // Values way outside [-1, 1] should clamp, not panic.
    let n = 1;
    let h = 4;
    let w = 4;
    let data: Vec<f64> = (0..(3 * h * w))
        .map(|i| if i % 2 == 0 { -1e6 } else { 1e6 })
        .collect();
    let x = arr(vec![n, 3, h, w], data);
    let svg = render_gallery(&x, None).expect("render gallery on extreme values");
    assert!(svg.contains("<rect"));
    // Saturated colors: rgb(0, ...) or rgb(255, ...) should be
    // present somewhere.
    assert!(
        svg.contains("rgb(0,0,0)")
            || svg.contains("rgb(255,255,255)")
            || svg.contains("rgb(255,0,255)")
    );
}

#[test]
fn dispatch_via_render_gallery_string() {
    // The svg() builtin dispatch routes "gallery" type to
    // render_gallery via the shared render(...) entry.
    let n = 2;
    let data: Vec<f64> = (0..(n * 3 * 4 * 4))
        .map(|i| (i as f64) / 100.0 - 1.0)
        .collect();
    let x = arr(vec![n, 3, 4, 4], data);
    let svg = render(&x, "gallery").expect("dispatch through render(...)");
    assert!(svg.starts_with("<svg"));
}

#[test]
fn gallery_overlay_one_value_per_thumbnail() {
    // [4, 3, 4, 4] images + [4] overlay of predictions.
    let n = 4;
    let imgs = arr(vec![n, 3, 4, 4], vec![0.0; n * 3 * 4 * 4]);
    let preds = arr(vec![n], vec![0.0, 1.0, 1.0, 0.0]);
    let svg = render_gallery(&imgs, Some(&preds)).expect("render gallery with overlay");
    assert!(svg.starts_with("<svg"));
    // Each thumbnail gets a <text> caption.
    let text_count = svg.matches("<text").count();
    assert_eq!(text_count, n, "expected {n} text labels, got {text_count}");
    // The label values should appear in the SVG.
    assert!(svg.contains(">0<") || svg.contains(">1<"));
}

#[test]
fn gallery_overlay_two_values_per_thumbnail_actual_predicted() {
    // [4, 3, 4, 4] images + [4, 2] overlay of actual/predicted.
    let n = 4;
    let imgs = arr(vec![n, 3, 4, 4], vec![0.0; n * 3 * 4 * 4]);
    let pairs = arr(
        vec![n, 2],
        vec![
            0.0, 0.0, // correct cat
            0.0, 1.0, // misclassified cat
            1.0, 1.0, // correct dog
            1.0, 0.0, // misclassified dog
        ],
    );
    let svg = render_gallery(&imgs, Some(&pairs)).expect("render gallery with [N, 2] overlay");
    // Each thumbnail gets one text caption containing
    // both values joined by '/'.
    assert!(svg.contains("0/1") || svg.contains("1/0"));
}

#[test]
fn gallery_overlay_shape_mismatch_errors() {
    let imgs = arr(vec![3, 3, 4, 4], vec![0.0; 3 * 3 * 4 * 4]);
    // Wrong N.
    let bad = arr(vec![4], vec![0.0; 4]);
    let err = render_gallery(&imgs, Some(&bad)).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
    // Wrong rank.
    let bad_rank = arr(vec![3, 2, 2], vec![0.0; 12]);
    let err = render_gallery(&imgs, Some(&bad_rank)).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
}

#[test]
fn attention_overlay_renders_multi_head_grid() {
    // 64x64 image; 4 heads x 16 patches.
    let img = arr(vec![3, 64, 64], vec![0.1; 3 * 64 * 64]);
    let attn: Vec<f64> = (0..64).map(|i| (i % 16) as f64 / 15.0).collect();
    let weights = arr(vec![4, 16], attn);
    let svg = render_attention_overlay(&img, Some(&weights)).expect("render overlay");
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
    // Overlay should add 4 * 16 = 64 translucent patch rects plus the
    // underlying image's pixel rects. Any translucent rect is enough.
    assert!(svg.contains("fill-opacity="));
}

#[test]
fn attention_overlay_dispatches_through_render_with_aux() {
    let img = arr(vec![3, 32, 32], vec![0.0; 3 * 32 * 32]);
    let attn = arr(vec![16], vec![0.5; 16]);
    let svg = render_with_aux(&img, "attention_overlay", Some(&attn)).expect("dispatch");
    assert!(svg.contains("fill-opacity="));
}

#[test]
fn attention_overlay_rejects_non_square_patch_count() {
    let img = arr(vec![3, 64, 64], vec![0.0; 3 * 64 * 64]);
    let attn = arr(vec![15], vec![0.0; 15]);
    let err = render_attention_overlay(&img, Some(&attn)).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
}

#[test]
fn attention_overlay_rejects_missing_aux() {
    let img = arr(vec![3, 16, 16], vec![0.0; 3 * 16 * 16]);
    let err = render_attention_overlay(&img, None).unwrap_err();
    assert!(matches!(err, VizError::InvalidShape(_)));
}
