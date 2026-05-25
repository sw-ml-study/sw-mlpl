//! Sandbox-root path resolver. Walks components manually so
//! `..` escapes are caught without touching the filesystem
//! (canonicalize would dereference symlinks; we don't want
//! that).

use std::path::{Component, Path, PathBuf};

use crate::error::LoaderHelperError;

pub fn resolve_in_sandbox(root: &Path, relative: &str) -> Result<PathBuf, LoaderHelperError> {
    let rel = Path::new(relative);
    if rel.is_absolute() {
        return Err(LoaderHelperError::AbsolutePathRejected {
            path: relative.into(),
            root: root.display().to_string(),
        });
    }
    check_components(rel, relative, root)?;
    Ok(root.join(rel))
}

fn check_components(rel: &Path, original: &str, root: &Path) -> Result<(), LoaderHelperError> {
    let mut depth: i64 = 0;
    for comp in rel.components() {
        match comp {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(sandbox_err(original, root));
                }
            }
            _ => return Err(rooted_err(original, root)),
        }
    }
    Ok(())
}

fn sandbox_err(original: &str, root: &Path) -> LoaderHelperError {
    LoaderHelperError::SandboxEscape {
        path: original.into(),
        root: root.display().to_string(),
    }
}

fn rooted_err(original: &str, root: &Path) -> LoaderHelperError {
    LoaderHelperError::RootedComponent {
        path: original.into(),
        root: root.display().to_string(),
    }
}
