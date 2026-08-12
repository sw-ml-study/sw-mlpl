//! Sandboxed filesystem I/O for the compile-to-Rust path. A compiled
//! binary has no interpreter `Environment`, so the sandbox root comes
//! from the process: the `MLPL_FS_ROOT` environment variable, else the
//! current working directory (the compiled analog of `--source-dir`).
//! Mirrors mlpl-eval's `read_bytes` / `read_range` / `file_size` /
//! `write_bytes` / `append_bytes` + the `contained` sandbox check, and
//! returns the same `ok(..)` / `err(..)` Results.
//!
//! This file is a FACADE (`mod` + `use` only). Behaviour lives in:
//! - `sandbox` -- the `contained` root check + numeric-arg validation.
//! - `read`    -- `read_bytes` (whole + range), `file_size`.
//! - `write`   -- `write_bytes`, `append_bytes`.

mod read;
mod sandbox;
mod write;

pub use read::{file_size, read_bytes, read_bytes_range};
pub use write::{append_bytes, write_bytes};
