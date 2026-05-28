//! Archive extraction + image-directory decode tests.

use std::fs;

use super::_test_helpers::{temp_dir, write_tiny_png};
use mlpl_eval_types::Value;

#[test]
fn extract_tarball_unpacks_files_into_dest() {
    let tmp = temp_dir("extract");
    let tar_path = tmp.join("test.tar.gz");
    let body = b"hi";
    let f = fs::File::create(&tar_path).unwrap();
    let gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
    let mut archive = tar::Builder::new(gz);
    let mut header = tar::Header::new_gnu();
    header.set_size(body.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(&mut header, "hello.txt", &body[..])
        .unwrap();
    archive.into_inner().unwrap().finish().unwrap();
    let dest = tmp.join("out");
    fs::create_dir_all(&dest).unwrap();
    crate::fetch_io::extract_tarball(&tar_path, &dest).unwrap();
    let got = fs::read_to_string(dest.join("hello.txt")).unwrap();
    assert_eq!(got, "hi");
}

#[test]
fn decode_directory_to_record_returns_expected_record() {
    let tmp = temp_dir("decode-dir");
    write_tiny_png(&tmp.join("Abyssinian_1.png"), [255, 0, 0]);
    write_tiny_png(&tmp.join("Bombay_1.png"), [0, 255, 0]);
    write_tiny_png(&tmp.join("beagle_1.png"), [0, 0, 255]);
    write_tiny_png(&tmp.join("pug_1.png"), [128, 128, 128]);
    let v = crate::fetch_io::decode_directory_to_record(&tmp, 4, 4).unwrap();
    let Value::Record { fields } = v else {
        panic!("expected Record");
    };
    assert_decode_fields(&fields);
}

fn assert_decode_fields(fields: &std::collections::BTreeMap<String, Value>) {
    let x = match fields.get("X") {
        Some(Value::Array(a)) => a,
        _ => panic!("X missing"),
    };
    assert_eq!(x.shape().dims(), &[4, 3, 4, 4]);
    let labels = x.labels().expect("X must carry axis labels");
    let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
    assert_eq!(names, vec!["batch", "channel", "y", "x"]);
    let y = match fields.get("Y") {
        Some(Value::Array(a)) => a,
        _ => panic!("Y missing"),
    };
    assert_eq!(y.data(), &[0.0, 0.0, 1.0, 1.0]);
    let names_list = match fields.get("names") {
        Some(Value::StrList { items }) => items,
        _ => panic!("names missing"),
    };
    assert_eq!(names_list.len(), 4);
    assert!(names_list[0].starts_with("Abyssinian"));
    assert!(names_list[3].starts_with("pug"));
}

#[test]
fn decode_directory_to_record_errors_on_empty_dir() {
    let tmp = temp_dir("decode-empty");
    let err = crate::fetch_io::decode_directory_to_record(&tmp, 4, 4).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("no PNG or JPEG"), "got: {msg}");
}
