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
            Component::Normal(_) | Component::CurDir => {
                depth += i64::from(matches!(comp, Component::Normal(_)));
            }
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(LoaderHelperError::SandboxEscape {
                        path: original.into(),
                        root: root.display().to_string(),
                    });
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(LoaderHelperError::RootedComponent {
                    path: original.into(),
                    root: root.display().to_string(),
                });
            }
        }
    }
    Ok(())
}
