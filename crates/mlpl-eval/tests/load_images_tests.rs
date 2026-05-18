//! Native `load_images(dir, [H, W])` and `image_io` decode
//! tests. Saga 29 step 003.
//!
//! All tests gated on the `image-io` feature -- the WASM
//! target stubs `load_images` with a clean error pointing
//! at `load_preloaded("pets_tiny")`.

#![cfg(feature = "image-io")]

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

const TINY_PNG_PATH: &str = "tests/data/tiny_4x4.png";

fn read_fixture_bytes() -> Vec<u8> {
    fs::read(TINY_PNG_PATH).unwrap_or_else(|e| panic!("read {TINY_PNG_PATH}: {e}"))
}

fn run_with_data_dir(src: &str, dir: &Path) -> Result<Value, mlpl_eval::EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    env.set_data_dir(dir.to_path_buf());
    eval_program_value(&stmts, &mut env)
}

#[test]
fn tiny_png_fixture_is_committed_and_decodable() {
    // The fixture is a 4x4 PNG built by
    // examples/build_tiny_fixtures.rs. If this fails the
    // fixture wasn't checked in; re-run the builder and
    // commit the file.
    let bytes = read_fixture_bytes();
    assert!(!bytes.is_empty(), "{TINY_PNG_PATH} is empty");
    // PNG magic: 89 50 4E 47.
    assert_eq!(
        &bytes[..4],
        &[0x89, 0x50, 0x4E, 0x47],
        "not a PNG (bad magic bytes)"
    );
}

#[test]
fn decode_and_resize_u8_identity_path() {
    // 4x4 -> 4x4: no actual resize, just decode-roundtrip
    // through bilinear (which preserves exact pixels when
    // dims match -- guaranteed by the unit test in
    // image_io::tests).
    let out = mlpl_eval::decode_and_resize_u8(Path::new(TINY_PNG_PATH), 4, 4)
        .expect("decode_and_resize_u8 on 4x4 PNG");
    assert_eq!(
        out.len(),
        4 * 4 * 3,
        "expected 48 RGB bytes, got {}",
        out.len()
    );
    // The committed fixture has red decreasing left-to-right.
    // First row: x = 0,1,2,3 -> red 255, 170, 85, 0.
    assert_eq!(out[0], 255, "top-left red should be 255, got {}", out[0]);
    let last_in_row = (3 * 3) as usize;
    assert_eq!(
        out[last_in_row], 0,
        "top-right red should be 0, got {}",
        out[last_in_row]
    );
}

#[test]
fn decode_and_resize_u8_downsamples_to_smaller() {
    let out =
        mlpl_eval::decode_and_resize_u8(Path::new(TINY_PNG_PATH), 2, 2).expect("decode 4x4 -> 2x2");
    assert_eq!(
        out.len(),
        2 * 2 * 3,
        "expected 12 RGB bytes, got {}",
        out.len()
    );
    // Top-left of the 2x2 should still be red-dominant; the
    // 4x4 fixture's top-left quadrant has high red values.
    assert!(
        out[0] >= 100,
        "top-left red should still be high, got {}",
        out[0]
    );
}

#[test]
fn load_images_with_no_data_dir_errors_with_pets_tiny_pointer() {
    let tokens = lex("load_images(\"some_dir\", [4, 4])").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let err = eval_program_value(&stmts, &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("load_preloaded"),
        "error should point at load_preloaded fallback, got: {msg}"
    );
}

#[test]
fn load_images_reads_a_single_image_directory() {
    // Set up a temp dir with our PNG fixture and ask
    // load_images to decode it. Use the project root as
    // sandbox so the fixture is reachable by relative path.
    let workspace_root = std::env::temp_dir().join("mlpl-load-images-tests");
    let temp = workspace_root.join("target/test-load-images");
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    // Copy the fixture in so the dir has exactly one PNG.
    let bytes = read_fixture_bytes();
    fs::write(temp.join("a.png"), &bytes).unwrap();
    // Use the parent as data_dir, "test-load-images" as the
    // relative path.
    let v = run_with_data_dir(
        "load_images(\"test-load-images\", [4, 4])",
        &workspace_root.join("target"),
    )
    .expect("load_images succeeded");
    match v {
        Value::Array(a) => {
            assert_eq!(a.shape().dims(), &[1, 3, 4, 4]);
            let labels = a.labels().expect("axis labels set");
            let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
            assert_eq!(names, vec!["batch", "channel", "y", "x"]);
            // Pixels are normalized to [-1, 1]: top-left red
            // (255) -> 1.0; top-right red (0) -> -1.0.
            // Channel 0 is red. data is row-major
            // [batch, channel, y, x] so index of (n=0, c=0,
            // y=0, x=0) = 0 and (0, 0, 0, 3) = 3.
            let data = a.data();
            assert!(
                (data[0] - 1.0).abs() < 1e-6,
                "top-left red should be 1.0, got {}",
                data[0]
            );
            assert!(
                (data[3] - (-1.0)).abs() < 1e-6,
                "top-right red should be -1.0, got {}",
                data[3]
            );
        }
        other => panic!("expected Value::Array, got {other:?}"),
    }
}

#[test]
fn load_images_errors_when_dir_has_no_images() {
    let workspace_root = std::env::temp_dir().join("mlpl-load-images-tests");
    let temp = workspace_root.join("target/test-empty-load-images");
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    let err = run_with_data_dir(
        "load_images(\"test-empty-load-images\", [4, 4])",
        &workspace_root.join("target"),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no PNG or JPEG"),
        "error should mention missing PNG/JPEG, got: {msg}"
    );
}

#[test]
fn load_images_rejects_bad_dim_args() {
    let workspace_root = std::env::temp_dir().join("mlpl-load-images-tests");
    let temp = workspace_root.join("target/test-load-images-baddim");
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    fs::write(temp.join("a.png"), read_fixture_bytes()).unwrap();
    // 1-element shape
    let err = run_with_data_dir(
        "load_images(\"test-load-images-baddim\", [4])",
        &workspace_root.join("target"),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("[H, W]"),
        "error should mention [H, W], got: {msg}"
    );
}

#[test]
fn load_images_returns_batch_axis_even_for_multiple_files() {
    let workspace_root = std::env::temp_dir().join("mlpl-load-images-tests");
    let temp = workspace_root.join("target/test-load-images-multi");
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    let bytes = read_fixture_bytes();
    fs::write(temp.join("a.png"), &bytes).unwrap();
    fs::write(temp.join("b.png"), &bytes).unwrap();
    fs::write(temp.join("c.png"), &bytes).unwrap();
    let v = run_with_data_dir(
        "load_images(\"test-load-images-multi\", [4, 4])",
        &workspace_root.join("target"),
    )
    .expect("load_images on 3-image dir");
    match v {
        Value::Array(a) => assert_eq!(a.shape().dims(), &[3, 3, 4, 4]),
        other => panic!("expected Value::Array, got {other:?}"),
    }
}

#[test]
fn load_images_filters_to_png_and_jpeg_extensions() {
    let workspace_root = std::env::temp_dir().join("mlpl-load-images-tests");
    let temp = workspace_root.join("target/test-load-images-filter");
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).unwrap();
    fs::write(temp.join("a.png"), read_fixture_bytes()).unwrap();
    // A non-image file in the same dir must not break decode.
    fs::write(temp.join("README.txt"), b"not an image").unwrap();
    let v = run_with_data_dir(
        "load_images(\"test-load-images-filter\", [4, 4])",
        &workspace_root.join("target"),
    )
    .expect("load_images skips non-images");
    match v {
        Value::Array(a) => assert_eq!(a.shape().dims(), &[1, 3, 4, 4]),
        other => panic!("expected Value::Array, got {other:?}"),
    }
    // Touch the HashSet import so clippy doesn't complain.
    let _ = HashSet::<&str>::new();
}
