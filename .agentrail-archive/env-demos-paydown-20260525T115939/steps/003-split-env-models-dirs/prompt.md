Saga 33 step 003: extract env_models.rs + env_dirs.rs from env.rs.

Continue env.rs split (now ~45 methods after step 002).

Move to crates/mlpl-eval/src/env_models.rs (impl Environment block):
- get_model, models_iter
- set_tokenizer, get_tokenizer, tokenizers_iter
(5 methods -- PASS)

Move to crates/mlpl-eval/src/env_dirs.rs (impl Environment block):
- set_data_dir, data_dir
- set_exp_dir, exp_dir
- push_experiment_log, experiment_log
(6 methods -- PASS)

Register both 'mod env_models;' and 'mod env_dirs;' in lib.rs.

Target: env.rs 45 -> 34 methods. Still FAIL but improving. Both new modules PASS individually.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1.