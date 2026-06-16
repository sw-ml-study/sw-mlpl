//! Tutorial lesson data. The PROSE + example code live in `lessons.toml`;
//! `build.rs` generates the `LESSONS` const from it into `OUT_DIR` (included
//! below), so content stays in data and out of Rust source. `Lesson` is the
//! type the generated const and all consumers reference.

pub struct Lesson {
    pub title: &'static str,
    pub intro: &'static str,
    pub examples: &'static [&'static str],
    pub try_it: &'static str,
}

// `pub const LESSONS: &[Lesson]`, generated from lessons.toml by build.rs.
include!(concat!(env!("OUT_DIR"), "/lessons_generated.rs"));
