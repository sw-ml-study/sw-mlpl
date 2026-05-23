//! `freeze(m)` / `unfreeze(m)` forward pass extracted from
//! `crates/mlpl-eval/src/model_freeze.rs`. Saga 33 step 011.
//!
//! Mark every parameter of a model as "frozen" so `adam` and
//! `momentum_sgd` skip the optimizer update for those names.
//! Gradients still flow through frozen parameters (the chain
//! rule does not care about freeze state); only the parameter
//! update is suppressed.
//!
//! C+D pattern in action:
//! - **Capability traits**: `freeze_inner` / `unfreeze_inner`
//!   take `&mut E: HasModels + HasFrozen`, never a concrete
//!   `Environment`.
//! - **Local error vocabulary**: `FreezeError` is owned by
//!   this crate; callers convert via their own
//!   `From<FreezeError>` impl.
//! - **Injected dependency**: the eval-loop fallback for
//!   non-ident-resolvable model args is passed as a closure,
//!   so this crate never imports the wider eval engine.

pub mod error;
pub mod inner;
pub mod resolve;

pub use error::FreezeError;
pub use inner::{freeze_inner, unfreeze_inner};
