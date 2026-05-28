//! Offline `pets_tiny.bin` builder. Saga 29 step 003.
//!
//! Reads the gitignored Oxford-IIIT Pet image checkout from
//! `data/oxford-iiit-pet/images/`, picks 100 cat + 100 dog
//! images, decodes + resizes each to 64x64 u8 RGB, and writes
//! `crates/mlpl-eval/data/pets_tiny.bin` in the wire format
//! that `src/pets_tiny.rs` expects.
//!
//! Convention: Oxford-IIIT Pet filenames distinguish cat
//! vs dog by case -- capitalized = cat breed
//! (`Abyssinian_1.jpg`), lowercase = dog breed
//! (`beagle_3.jpg`). The label byte is `0` for cat and `1`
//! for dog so a binary classifier targets `Y` directly.
//!
//! Build + run with the `image-io` feature enabled:
//!
//! ```bash
//! cargo run --example build_pets_tiny \
//!   --features image-io -p mlpl-eval
//! ```
//!
//! The committed `pets_tiny.bin` lands at
//! `crates/mlpl-eval/data/pets_tiny.bin` and is shipped via
//! `include_bytes!` -- after running this, `git add` and
//! commit the new fixture in the same step that wires
//! `load_preloaded("pets_tiny")` end-to-end.

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "build_pets_tiny requires the `image-io` feature; rebuild with \
         --features image-io"
    );
    std::process::exit(1);
}

#[cfg(feature = "image-io")]
use std::path::PathBuf;
#[cfg(feature = "image-io")]
use std::{env, fs, io::Write, process};

#[cfg(feature = "image-io")]
use mlpl_eval::decode_and_resize_u8;

#[cfg(feature = "image-io")]
const TARGET_SIZE: usize = 64;
#[cfg(feature = "image-io")]
const CATS_PER_CLASS: usize = 100;
#[cfg(feature = "image-io")]
const DOGS_PER_CLASS: usize = 100;
#[cfg(feature = "image-io")]
const MAGIC: &[u8; 8] = b"MLPLPETS";
#[cfg(feature = "image-io")]
const VERSION: u32 = 1;

#[cfg(feature = "image-io")]
fn main() {
    if let Err(e) = run() {
        eprintln!("build_pets_tiny: {e}");
        process::exit(1);
    }
}

#[cfg(feature = "image-io")]
fn run() -> Result<(), String> {
    let workspace_root = workspace_root()?;
    let images_dir = workspace_root.join("data/oxford-iiit-pet/images");
    if !images_dir.is_dir() {
        return Err(format!(
            "{} not found -- extract the Oxford-IIIT Pet tarball there first",
            images_dir.display()
        ));
    }
    let out_path = workspace_root.join("crates/mlpl-eval/data/pets_tiny.bin");

    let (cats, dogs) = pick_files(&images_dir)?;
    if cats.len() < CATS_PER_CLASS {
        return Err(format!(
            "only {} cat files in {}, need {}",
            cats.len(),
            images_dir.display(),
            CATS_PER_CLASS
        ));
    }
    if dogs.len() < DOGS_PER_CLASS {
        return Err(format!(
            "only {} dog files in {}, need {}",
            dogs.len(),
            images_dir.display(),
            DOGS_PER_CLASS
        ));
    }

    let mut picks: Vec<(PathBuf, u8)> = Vec::with_capacity(CATS_PER_CLASS + DOGS_PER_CLASS);
    for p in cats.iter().take(CATS_PER_CLASS) {
        picks.push((p.clone(), 0));
    }
    for p in dogs.iter().take(DOGS_PER_CLASS) {
        picks.push((p.clone(), 1));
    }

    let mut y_bytes = Vec::with_capacity(picks.len());
    let mut names_buf: Vec<u8> = Vec::new();
    let mut x_bytes: Vec<u8> = Vec::with_capacity(picks.len() * 3 * TARGET_SIZE * TARGET_SIZE);
    for (idx, (path, label)) in picks.iter().enumerate() {
        let rgb_hwc = decode_and_resize_u8(path, TARGET_SIZE, TARGET_SIZE)
            .map_err(|e| format!("[{idx}] decode/resize {}: {e}", path.display()))?;
        if rgb_hwc.len() != 3 * TARGET_SIZE * TARGET_SIZE {
            return Err(format!(
                "[{idx}] {} resized to wrong size: got {} bytes",
                path.display(),
                rgb_hwc.len()
            ));
        }
        // HWC -> CHW
        let mut chw = vec![0u8; rgb_hwc.len()];
        for c in 0..3 {
            for y in 0..TARGET_SIZE {
                for x in 0..TARGET_SIZE {
                    let src = (y * TARGET_SIZE + x) * 3 + c;
                    let dst = c * TARGET_SIZE * TARGET_SIZE + y * TARGET_SIZE + x;
                    chw[dst] = rgb_hwc[src];
                }
            }
        }
        x_bytes.extend_from_slice(&chw);
        y_bytes.push(*label);
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("[{idx}] non-utf8 filename"))?;
        names_buf.extend_from_slice(name.as_bytes());
        names_buf.push(0);
        if idx % 20 == 0 {
            println!("  {idx:>3} / {} ({})", picks.len(), name);
        }
    }
    let n = picks.len();
    let names_len =
        u32::try_from(names_buf.len()).map_err(|_| "name table exceeds u32".to_string())?;

    let mut out = Vec::with_capacity(28 + n + 4 + names_buf.len() + x_bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(TARGET_SIZE as u32).to_le_bytes());
    out.extend_from_slice(&(TARGET_SIZE as u32).to_le_bytes());
    out.extend_from_slice(&y_bytes);
    out.extend_from_slice(&names_len.to_le_bytes());
    out.extend_from_slice(&names_buf);
    out.extend_from_slice(&x_bytes);

    fs::create_dir_all(out_path.parent().unwrap()).map_err(|e| format!("mkdir: {e}"))?;
    let mut f = fs::File::create(&out_path).map_err(|e| format!("create: {e}"))?;
    f.write_all(&out).map_err(|e| format!("write: {e}"))?;
    println!("wrote {} ({} bytes)", out_path.display(), out.len());
    Ok(())
}

#[cfg(feature = "image-io")]
fn pick_files(images_dir: &std::path::Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
    use std::collections::BTreeMap;
    let mut all: Vec<PathBuf> = fs::read_dir(images_dir)
        .map_err(|e| format!("readdir {}: {e}", images_dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
                .unwrap_or(false)
        })
        .collect();
    all.sort();
    // Group by breed (filename prefix up to the last `_`).
    // Breeds with a leading capital are cats; lowercase are dogs.
    let mut by_breed: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for p in all {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let breed = name
            .rsplit_once('_')
            .map(|(b, _)| b.to_string())
            .unwrap_or_else(|| name.to_string());
        by_breed.entry(breed).or_default().push(p);
    }
    let mut cat_breeds: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut dog_breeds: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for (breed, files) in by_breed {
        let first = breed.chars().next().unwrap_or(' ');
        if first.is_ascii_uppercase() {
            cat_breeds.insert(breed, files);
        } else if first.is_ascii_lowercase() {
            dog_breeds.insert(breed, files);
        }
    }
    // Round-robin pick across breeds so the 100 per class
    // are spread evenly: one from each breed, then the
    // second from each, until we have enough.
    let cats = round_robin(&cat_breeds, CATS_PER_CLASS);
    let dogs = round_robin(&dog_breeds, DOGS_PER_CLASS);
    Ok((cats, dogs))
}

#[cfg(feature = "image-io")]
fn round_robin(
    breeds: &std::collections::BTreeMap<String, Vec<PathBuf>>,
    want: usize,
) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::with_capacity(want);
    let mut indices: Vec<usize> = vec![0; breeds.len()];
    loop {
        let mut progress = false;
        for (i, (_breed, files)) in breeds.iter().enumerate() {
            if out.len() >= want {
                return out;
            }
            if indices[i] < files.len() {
                out.push(files[indices[i]].clone());
                indices[i] += 1;
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    out
}

#[cfg(feature = "image-io")]
fn workspace_root() -> Result<PathBuf, String> {
    // CARGO_MANIFEST_DIR points at crates/mlpl-eval/.
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())?;
    let p = PathBuf::from(manifest_dir);
    // workspace root is two levels up: crates/mlpl-eval -> crates -> root
    p.parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or_else(|| "cannot resolve workspace root".to_string())
}
