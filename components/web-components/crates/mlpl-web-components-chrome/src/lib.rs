//! Page-chrome Yew components for the web playground:
//! header, footer, and the GithubCorner badge. Extracted from
//! mlpl-web during saga 82. All three only require yew.
//!
//! mlpl-web re-exports these symbols under `crate::components::*`
//! so existing call sites stay unchanged.

pub mod footer;
pub mod github_corner;
pub mod header;
