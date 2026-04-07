//! SVG output handling for the CLI REPL.

use std::path::PathBuf;

/// Where to write SVG output produced by the REPL.
pub struct SvgOut {
    pub dir: Option<PathBuf>,
    pub counter: u32,
}

impl SvgOut {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self { dir, counter: 0 }
    }

    /// Handle a single SVG result -- either save to disk or print a placeholder.
    pub fn handle(&mut self, svg: &str) {
        if let Some(dir) = &self.dir {
            self.counter += 1;
            let path = dir.join(format!("svg-{:03}.svg", self.counter));
            match std::fs::write(&path, svg) {
                Ok(()) => println!("[svg: {} bytes -> {}]", svg.len(), path.display()),
                Err(e) => eprintln!("[svg: error writing {}: {e}]", path.display()),
            }
        } else {
            println!("[svg: {} bytes -- pass --svg-out <dir> to save]", svg.len());
        }
    }
}
