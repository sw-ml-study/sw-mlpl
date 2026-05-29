//! Image-upload helpers + `EvalDeps` (the Yew state bundle
//! shared by every REPL-eval handler). EvalDeps lives here
//! because both upload_cmd and submit (in handlers-eval) need
//! it, and putting it here keeps the dep DAG clean
//! (handlers-eval -> handlers-upload, no cycle).

pub mod eval_deps;
pub mod upload;
pub mod upload_cmd;
