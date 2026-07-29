//! Environment USER-FUNCTION capabilities as traits (eval
//! decomposition capability peel): define/look up `u:` functions
//! (`EnvUserFns`) and render their `:fns`/`:describe`/`:list`
//! listings (`EnvUserFnsRender`). The `UserFn` type lives in
//! mlpl-eval-state (it is an `Environment` field type).

pub mod env_user_fns;
pub mod env_user_fns_render;

pub use env_user_fns::EnvUserFns;
pub use env_user_fns_render::EnvUserFnsRender;
