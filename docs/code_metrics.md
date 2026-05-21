## Code Metric Gates and Architecture Guide

This section guides AI coding agents in creating loosely-coupled, functional-style, testable, and maintainable Rust code that fits small-part complexity gates. The principles apply broadly across languages; the examples are Rust.

### Target Metric Gates

| Level | Preferred Gate |
| --- | --- |
| Lines per function | ≤ 25 LOC |
| Functions per module | ≤ 5 |
| Modules per crate | ≤ 5 |
| Crates per component/workspace group | ≤ 5 |
| Major subsystems per repo | ≤ 5-ish |

**Guiding rule:** Every part should fit in human working memory. When a part grows, split by responsibility, not by accident. The 25-LOC function gate keeps a function on a single page view. The 5-per gate (mental load 5 ± 2) reserves room for future expansion.

---

### 1. Core Design Principles

#### 1.1 Prefer small pure functions

Prefer:

```rust
fn normalize_name(input: &str) -> String {
    input.trim().to_ascii_lowercase()
}
```

Over:

```rust
impl AppState {
    fn normalize_name_for_current_context(&self, input: &str) -> String {
        // reads state, logs, mutates cache, validates, normalizes...
    }
}
```

Use pure functions for: parsing, validation, transformation, formatting, decision logic, classification, filtering, mapping, scoring.

Keep `impl` methods mostly for: constructors, state mutation, trait implementations, small orchestration methods, invariant-preserving operations.

#### 1.2 Separate decisions from effects

Bad:

```rust
fn process_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let config = parse_config(&text)?;
    if config.enabled {
        fs::write("out.txt", render(config)?)?;
    }
    Ok(())
}
```

Better:

```rust
fn plan_output(config: &Config) -> Option<OutputPlan> {
    config.enabled.then(|| OutputPlan::from(config))
}

fn process_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let config = parse_config(&text)?;
    if let Some(plan) = plan_output(&config) {
        fs::write("out.txt", render_output(&plan)?)?;
    }
    Ok(())
}
```

**Rule:** Pure code decides. Thin shell code performs effects. This makes tests smaller and avoids mock-heavy designs.

---

### 2. Refactoring Triggers

When a function exceeds 25 LOC, do not split at random. Diagnose, then split by responsibility:

| Symptom | Refactor |
| --- | --- |
| Function parses and validates | Extract `parse_*` and `validate_*` |
| Function builds config and runs app | Extract `load_config`, `build_context`, `run_command` |
| Function has many if/else branches | Extract decision table, enum dispatch, or strategy |
| Function mutates several fields | Extract state transition function |
| Function handles errors inline | Extract helper returning typed error/result |
| Function has setup, action, cleanup | Extract template-hook or RAII guard |
| Function loops with complex body | Extract loop body into named function |
| Function has nested matches | Extract per-variant handler |

---

### 3. Preferred Module Shape

Each module should have one clear job:

```
src/
  lib.rs
  config/
    mod.rs
    parse.rs
    validate.rs
    defaults.rs
  command/
    mod.rs
    args.rs
    plan.rs
    run.rs
  model/
    mod.rs
    component.rs
    metric.rs
    report.rs
  report/
    mod.rs
    markdown.rs
    json.rs
    summary.rs
```

Inside each `mod.rs`:

```rust
mod parse;
mod validate;
mod defaults;

pub use parse::parse_config;
pub use validate::validate_config;
pub use defaults::default_config;
```

**Rule:** `mod.rs` is a facade, not a junk drawer.

---

### 4. File / Module Roles

Use consistent names so agents know where code belongs:

| File | Purpose |
| --- | --- |
| `model.rs` | Data structures and domain types |
| `parse.rs` | String/file/input → typed data |
| `validate.rs` | Typed data → validation result |
| `plan.rs` | Inputs/config → execution plan |
| `run.rs` | Performs effects |
| `render.rs` | Typed data → string/output |
| `error.rs` | Error enums and conversions |
| `test_support.rs` | Shared test builders/helpers |
| `fixtures.rs` | Static test data or fixture loading |

---

### 5. Keep Tests Out of Production Modules

Prefer either a sibling tests directory or a sibling test file:

```
src/
  config/
    parse.rs
tests/
  config_parse_tests.rs
```

Or:

```
src/
  config/
    parse.rs
    parse_tests.rs
```

Avoid giant inline test blocks that make source files unreadable. Production files should stay production-focused.

---

### 6. Composition Over God Objects

Avoid one struct that owns everything:

```rust
struct App {
    config: Config,
    db: Db,
    logger: Logger,
    parser: Parser,
    renderer: Renderer,
    state: State,
}
```

Prefer smaller parts:

```rust
struct Runtime {
    config: Config,
    services: Services,
}

struct Services {
    store: Box<dyn Store>,
    renderer: Box<dyn Renderer>,
}
```

Or pass capabilities directly:

```rust
fn run_report(
    input: &Input,
    store: &dyn Store,
    renderer: &dyn Renderer,
) -> Result<Report> {
    let data = store.load(input)?;
    renderer.render(&data)
}
```

**Rule:** Pass the smallest capability needed.

---

### 7. Trait Guidance

Use traits for boundaries, not for every helper.

Good trait candidates: file system abstraction, network/API client, clock/time source, renderer backend, storage backend, command handler, plugin/extension point.

Avoid traits for: simple pure functions, one implementation with no expected variants, internal helpers, premature abstraction.

```rust
pub trait Store {
    fn load(&self, key: &str) -> Result<Record>;
}

pub fn load_report(store: &dyn Store, key: &str) -> Result<Report> {
    let record = store.load(key)?;
    Ok(Report::from(record))
}
```

---

### 8. Pattern Playbook

#### Facade

Use when a subsystem has many internal modules:

```rust
pub fn analyze_repo(path: &Path) -> Result<RepoReport> {
    let crates = discover_crates(path)?;
    let metrics = measure_crates(&crates)?;
    summarize_repo(metrics)
}
```

#### Builder

Use when construction has many optional fields:

```rust
let config = ConfigBuilder::new()
    .max_fn_loc(25)
    .max_module_fns(5)
    .build()?;
```

Keep builder methods tiny.

#### Chain of Responsibility

Use when several handlers may process something. Good for metric gates, lint rules, validators, and policy checks:

```rust
for rule in rules {
    if let Some(issue) = rule.check(item) {
        issues.push(issue);
    }
}
```

#### Bridge

Use when policy and backend vary independently. Example: metric policy (strict / relaxed / experimental) crossed with output backend (markdown / json / terminal). Do not combine these into one giant type.

#### Template-Hook

Use when an algorithm is fixed but some steps vary:

```rust
fn run_analysis<H: Hooks>(hooks: &H, input: Input) -> Result<Report> {
    hooks.before_parse(&input)?;
    let parsed = parse(input)?;
    hooks.after_parse(&parsed)?;
    analyze(parsed)
}
```

---

### 9. Error Handling Style

Prefer local, typed errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing field: {0}")]
    MissingField(&'static str),

    #[error("invalid limit: {0}")]
    InvalidLimit(usize),
}
```

Avoid giant global error enums unless the crate is very small. Use a focused `type Result<T> = std::result::Result<T, ConfigError>;` inside each module.

---

### 10. Agent Refactoring Algorithm

When modifying code, follow this loop:

1. **Measure** — function LOC, function count per module, module count per crate, crate count per component, coupling/import size, test size and placement.
2. **Classify excess** — too many LOC? too many functions? too many responsibilities? too many effects mixed with logic? too many variants in one match? too many tests inline?
3. **Split by responsibility** — prefer names like `parse_*`, `validate_*`, `build_*`, `plan_*`, `run_*`, `render_*`, `summarize_*`.
4. **Preserve behavior** — run `cargo test` before refactoring, and after each small move run `cargo test` and `cargo clippy --all-targets --all-features`.
5. **Re-measure** — the refactor is not complete until the metrics pass.

---

### 11. Concrete Agent Rules

- No function over 25 LOC unless explicitly justified.
- No module with more than 5 production functions.
- Move tests out of production files when they distort readability.
- Prefer pure free functions for logic.
- Use `impl` for invariants, construction, and behavior tied to state.
- Separate config, initialization, runtime state, and command handling.
- Separate parsing from validation.
- Separate planning from execution.
- Separate rendering from data modeling.
- Use facades to hide subsystem detail.
- Use traits only at real boundaries.
- Never create a god struct, god enum, god module, or god crate.

---

### 12. Suggested Repository Shape

```
repo/
  Cargo.toml
  crates/
    cli/
    core/
    report/
    rules/
    test-support/
  docs/
    architecture.md
    design.md
    metrics.md
  tests/
    cli_tests.rs
```

Possible crate roles:

| Crate | Purpose |
| --- | --- |
| `core` | Domain model and pure analysis |
| `rules` | Metric gates and validation policies |
| `report` | Markdown/JSON/terminal output |
| `cli` | Arg parsing, config loading, effectful shell |
| `test-support` | Fixtures and helpers |

---

### 13. `lib.rs` and `mod.rs` Are Facades Only

Do not put executable logic in `src/lib.rs`, `src/main.rs`, or any `src/foo/mod.rs`.

Allowed in `lib.rs` / `mod.rs`:

```rust
pub mod config;
pub mod rules;
pub mod report;

pub use config::Config;
pub use rules::RuleSet;
```

Not allowed:

```rust
pub fn analyze_repo(...) -> Result<Report> {
    ...
}
```

Instead:

```
src/
  lib.rs
  analysis/
    mod.rs
    analyze.rs
    plan.rs
```

```rust
// analysis/mod.rs
mod analyze;
mod plan;

pub use analyze::analyze_repo;
```

```rust
// analysis/analyze.rs
pub fn analyze_repo(...) -> Result<Report> {
    ...
}
```

**Rule:** `lib.rs` and `mod.rs` define the public surface. Named files contain behavior.

---

### 14. `build.rs` Follows the Same Rules

Treat `build.rs` as a tiny compile-time CLI.

Bad:

```rust
// build.rs
fn main() {
    // 200 lines of file discovery, parsing, codegen, env handling...
}
```

Better:

```
build.rs
build_support/
  mod.rs
  env.rs
  discover.rs
  generate.rs
  rerun.rs
  run.rs
```

```rust
// build.rs
mod build_support;

fn main() {
    build_support::run();
}
```

```rust
// build_support/mod.rs
mod discover;
mod env;
mod generate;
mod rerun;
mod run;

pub use run::run;
```

```rust
// build_support/run.rs
pub fn run() {
    rerun::emit_rerun_directives();
    let inputs = discover::find_inputs();
    generate::write_generated_code(inputs);
}
```

For build scripts:

| Concern | Module |
| --- | --- |
| `cargo:rerun-if-changed` | `rerun.rs` |
| Env vars | `env.rs` |
| Input discovery | `discover.rs` |
| Code generation | `generate.rs` |
| Output paths | `paths.rs` |
| Error handling | `error.rs` |

**Rule:** Build-time code deserves the same maintainability as runtime code.

---

### 15. Macro Guidance

Use macros to remove repetition, not to hide design.

**Good macro use:** repetitive trait impls, repetitive test cases, boilerplate enum conversions, static rule declarations, DSL-like configuration, generated table rows, repeated parser cases.

**Bad macro use:** hiding complex control flow, avoiding proper functions, encoding business logic in token soup, making errors harder to understand, replacing simple generics or helper functions.

#### Macro Types and Best Uses

| Macro Type | Use For | Avoid For |
| --- | --- | --- |
| `macro_rules!` | Small repetitive patterns | Complex parsing |
| Derive proc macro | Boilerplate trait impls | Behavior-heavy logic |
| Attribute proc macro | Declarative annotations | Hidden runtime effects |
| Function-like proc macro | DSLs/codegen | Ordinary helper logic |
| Build-time codegen | Large generated tables/types | Small hand-written code |

#### Prefer this order before writing a macro

1. A pure function
2. A generic function
3. A trait
4. A builder
5. A table-driven design
6. A small `macro_rules!`
7. A proc macro
8. Build-time codegen

**Rule:** Macros are for structural repetition, not unclear abstraction.

#### Keep macros small and isolated

```
src/
  lib.rs
  macros/
    mod.rs
    declare_rule.rs
    table_tests.rs
```

```rust
// macros/mod.rs
mod declare_rule;
mod table_tests;

pub(crate) use declare_rule::declare_rule;
pub(crate) use table_tests::table_tests;
```

A macro file should usually contain: one macro, one responsibility, documentation, an example expansion or usage, tests where practical.

#### Example: Macro for repetitive tests

```rust
macro_rules! metric_case {
    ($name:ident, $metric:expr, $value:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let result = check_metric($metric, $value);
            assert_eq!(result.is_ok(), $expected);
        }
    };
}

metric_case!(accepts_25_loc_fn, "fn_loc", 25, true);
metric_case!(rejects_26_loc_fn, "fn_loc", 26, false);
```

#### Example: Macro for rule declaration

```rust
macro_rules! declare_limit_rule {
    ($name:ident, $metric:expr, $limit:expr) => {
        pub fn $name(value: usize) -> RuleResult {
            check_limit($metric, value, $limit)
        }
    };
}

declare_limit_rule!(check_fn_loc, Metric::FunctionLoc, 25);
declare_limit_rule!(check_module_fns, Metric::ModuleFunctions, 5);
```

The macro expands structure; the function contains behavior:

```rust
pub fn check_limit(metric: Metric, value: usize, limit: usize) -> RuleResult {
    if value <= limit {
        RuleResult::Pass
    } else {
        RuleResult::Fail { metric, value, limit }
    }
}
```

#### Proc macro crate shape

If proc macros are needed, isolate them in their own crate:

```
crates/
  metric-core/
  metric-macros/
  metric-cli/
```

```toml
# crates/metric-macros/Cargo.toml
[lib]
proc-macro = true
```

Suggested layout:

```
metric-macros/
  src/
    lib.rs
    derive_rule.rs
    attr_metric.rs
    parse.rs
    emit.rs
    error.rs
```

`lib.rs` should only expose macro entry points:

```rust
#[proc_macro_derive(Rule)]
pub fn derive_rule(input: TokenStream) -> TokenStream {
    derive_rule::expand(input)
}
```

Actual behavior belongs in named files.

---

### 16. Final Agent Directive

Refactor and write Rust code according to these gates and layout rules:

**Metrics**
- max 25 LOC per function
- max 5 production functions per module
- max 5 modules per crate where practical
- keep repo/component/subsystem hierarchy shallow and grouped

**Layout**
- do not put functions in `lib.rs`
- do not put functions in `mod.rs`
- `lib.rs` and `mod.rs` are facade/re-export files only
- place behavior in named module files
- move tests out of production modules when they distort readability

**Design**
- prefer pure free functions for logic
- separate parsing, validation, planning, execution, rendering, and state
- separate configuration, initialization, runtime state, and arg parsing
- use traits only at real boundaries
- use facades for subsystem entry points
- use composition over god structs
- use design patterns where they reduce coupling

**Build scripts**
- `build.rs` must remain a thin entry point
- split compile-time logic into named modules
- apply the same LOC/function/module rules to build-time code

**Macros**
- use macros to remove boilerplate and repetitive structure
- do not use macros to hide complex logic
- prefer functions/generics/traits before macros
- keep macro definitions small and isolated
- `macro_rules!` is preferred for simple repetition
- proc macros require their own crate
- macro expansion should call ordinary functions where behavior lives

**Process**
- measure before refactoring
- classify the excess responsibility
- split by responsibility, not mechanically
- preserve behavior
- run tests after each coherent change
- re-measure before declaring success

**Stricter rule:** Do not add new logic to an over-limit function or module. First extract responsibilities into named pure helpers, move tests out of production modules, and re-run the metric tool.

---


---

**Added by Proact on 2026-05-21 10:08:18**

## Code Metric Gates and Architecture Guide

This section guides AI coding agents in creating loosely-coupled, functional-style, testable, and maintainable Rust code that fits small-part complexity gates. The principles apply broadly across languages; the examples are Rust.

### Target Metric Gates

| Level | Preferred Gate |
| --- | --- |
| Lines per function | ≤ 25 LOC |
| Functions per module | ≤ 5 |
| Modules per crate | ≤ 5 |
| Crates per component/workspace group | ≤ 5 |
| Major subsystems per repo | ≤ 5-ish |

**Guiding rule:** Every part should fit in human working memory. When a part grows, split by responsibility, not by accident. The 25-LOC function gate keeps a function on a single page view. The 5-per gate (mental load 5 ± 2) reserves room for future expansion.

---

### 1. Core Design Principles

#### 1.1 Prefer small pure functions

Prefer:

```rust
fn normalize_name(input: &str) -> String {
    input.trim().to_ascii_lowercase()
}
```

Over:

```rust
impl AppState {
    fn normalize_name_for_current_context(&self, input: &str) -> String {
        // reads state, logs, mutates cache, validates, normalizes...
    }
}
```

Use pure functions for: parsing, validation, transformation, formatting, decision logic, classification, filtering, mapping, scoring.

Keep `impl` methods mostly for: constructors, state mutation, trait implementations, small orchestration methods, invariant-preserving operations.

#### 1.2 Separate decisions from effects

Bad:

```rust
fn process_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let config = parse_config(&text)?;
    if config.enabled {
        fs::write("out.txt", render(config)?)?;
    }
    Ok(())
}
```

Better:

```rust
fn plan_output(config: &Config) -> Option<OutputPlan> {
    config.enabled.then(|| OutputPlan::from(config))
}

fn process_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let config = parse_config(&text)?;
    if let Some(plan) = plan_output(&config) {
        fs::write("out.txt", render_output(&plan)?)?;
    }
    Ok(())
}
```

**Rule:** Pure code decides. Thin shell code performs effects. This makes tests smaller and avoids mock-heavy designs.

---

### 2. Refactoring Triggers

When a function exceeds 25 LOC, do not split at random. Diagnose, then split by responsibility:

| Symptom | Refactor |
| --- | --- |
| Function parses and validates | Extract `parse_*` and `validate_*` |
| Function builds config and runs app | Extract `load_config`, `build_context`, `run_command` |
| Function has many if/else branches | Extract decision table, enum dispatch, or strategy |
| Function mutates several fields | Extract state transition function |
| Function handles errors inline | Extract helper returning typed error/result |
| Function has setup, action, cleanup | Extract template-hook or RAII guard |
| Function loops with complex body | Extract loop body into named function |
| Function has nested matches | Extract per-variant handler |

---

### 3. Preferred Module Shape

Each module should have one clear job:

```
src/
  lib.rs
  config/
    mod.rs
    parse.rs
    validate.rs
    defaults.rs
  command/
    mod.rs
    args.rs
    plan.rs
    run.rs
  model/
    mod.rs
    component.rs
    metric.rs
    report.rs
  report/
    mod.rs
    markdown.rs
    json.rs
    summary.rs
```

Inside each `mod.rs`:

```rust
mod parse;
mod validate;
mod defaults;

pub use parse::parse_config;
pub use validate::validate_config;
pub use defaults::default_config;
```

**Rule:** `mod.rs` is a facade, not a junk drawer.

---

### 4. File / Module Roles

Use consistent names so agents know where code belongs:

| File | Purpose |
| --- | --- |
| `model.rs` | Data structures and domain types |
| `parse.rs` | String/file/input → typed data |
| `validate.rs` | Typed data → validation result |
| `plan.rs` | Inputs/config → execution plan |
| `run.rs` | Performs effects |
| `render.rs` | Typed data → string/output |
| `error.rs` | Error enums and conversions |
| `test_support.rs` | Shared test builders/helpers |
| `fixtures.rs` | Static test data or fixture loading |

---

### 5. Keep Tests Out of Production Modules

Prefer either a sibling tests directory or a sibling test file:

```
src/
  config/
    parse.rs
tests/
  config_parse_tests.rs
```

Or:

```
src/
  config/
    parse.rs
    parse_tests.rs
```

Avoid giant inline test blocks that make source files unreadable. Production files should stay production-focused.

---

### 6. Composition Over God Objects

Avoid one struct that owns everything:

```rust
struct App {
    config: Config,
    db: Db,
    logger: Logger,
    parser: Parser,
    renderer: Renderer,
    state: State,
}
```

Prefer smaller parts:

```rust
struct Runtime {
    config: Config,
    services: Services,
}

struct Services {
    store: Box<dyn Store>,
    renderer: Box<dyn Renderer>,
}
```

Or pass capabilities directly:

```rust
fn run_report(
    input: &Input,
    store: &dyn Store,
    renderer: &dyn Renderer,
) -> Result<Report> {
    let data = store.load(input)?;
    renderer.render(&data)
}
```

**Rule:** Pass the smallest capability needed.

---

### 7. Trait Guidance

Use traits for boundaries, not for every helper.

Good trait candidates: file system abstraction, network/API client, clock/time source, renderer backend, storage backend, command handler, plugin/extension point.

Avoid traits for: simple pure functions, one implementation with no expected variants, internal helpers, premature abstraction.

```rust
pub trait Store {
    fn load(&self, key: &str) -> Result<Record>;
}

pub fn load_report(store: &dyn Store, key: &str) -> Result<Report> {
    let record = store.load(key)?;
    Ok(Report::from(record))
}
```

---

### 8. Pattern Playbook

#### Facade

Use when a subsystem has many internal modules:

```rust
pub fn analyze_repo(path: &Path) -> Result<RepoReport> {
    let crates = discover_crates(path)?;
    let metrics = measure_crates(&crates)?;
    summarize_repo(metrics)
}
```

#### Builder

Use when construction has many optional fields:

```rust
let config = ConfigBuilder::new()
    .max_fn_loc(25)
    .max_module_fns(5)
    .build()?;
```

Keep builder methods tiny.

#### Chain of Responsibility

Use when several handlers may process something. Good for metric gates, lint rules, validators, and policy checks:

```rust
for rule in rules {
    if let Some(issue) = rule.check(item) {
        issues.push(issue);
    }
}
```

#### Bridge

Use when policy and backend vary independently. Example: metric policy (strict / relaxed / experimental) crossed with output backend (markdown / json / terminal). Do not combine these into one giant type.

#### Template-Hook

Use when an algorithm is fixed but some steps vary:

```rust
fn run_analysis<H: Hooks>(hooks: &H, input: Input) -> Result<Report> {
    hooks.before_parse(&input)?;
    let parsed = parse(input)?;
    hooks.after_parse(&parsed)?;
    analyze(parsed)
}
```

---

### 9. Error Handling Style

Prefer local, typed errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("missing field: {0}")]
    MissingField(&'static str),

    #[error("invalid limit: {0}")]
    InvalidLimit(usize),
}
```

Avoid giant global error enums unless the crate is very small. Use a focused `type Result<T> = std::result::Result<T, ConfigError>;` inside each module.

---

### 10. Agent Refactoring Algorithm

When modifying code, follow this loop:

1. **Measure** — function LOC, function count per module, module count per crate, crate count per component, coupling/import size, test size and placement.
2. **Classify excess** — too many LOC? too many functions? too many responsibilities? too many effects mixed with logic? too many variants in one match? too many tests inline?
3. **Split by responsibility** — prefer names like `parse_*`, `validate_*`, `build_*`, `plan_*`, `run_*`, `render_*`, `summarize_*`.
4. **Preserve behavior** — run `cargo test` before refactoring, and after each small move run `cargo test` and `cargo clippy --all-targets --all-features`.
5. **Re-measure** — the refactor is not complete until the metrics pass.

---

### 11. Concrete Agent Rules

- No function over 25 LOC unless explicitly justified.
- No module with more than 5 production functions.
- Move tests out of production files when they distort readability.
- Prefer pure free functions for logic.
- Use `impl` for invariants, construction, and behavior tied to state.
- Separate config, initialization, runtime state, and command handling.
- Separate parsing from validation.
- Separate planning from execution.
- Separate rendering from data modeling.
- Use facades to hide subsystem detail.
- Use traits only at real boundaries.
- Never create a god struct, god enum, god module, or god crate.

---

### 12. Suggested Repository Shape

```
repo/
  Cargo.toml
  crates/
    cli/
    core/
    report/
    rules/
    test-support/
  docs/
    architecture.md
    design.md
    metrics.md
  tests/
    cli_tests.rs
```

Possible crate roles:

| Crate | Purpose |
| --- | --- |
| `core` | Domain model and pure analysis |
| `rules` | Metric gates and validation policies |
| `report` | Markdown/JSON/terminal output |
| `cli` | Arg parsing, config loading, effectful shell |
| `test-support` | Fixtures and helpers |

---

### 13. `lib.rs` and `mod.rs` Are Facades Only

Do not put executable logic in `src/lib.rs`, `src/main.rs`, or any `src/foo/mod.rs`.

Allowed in `lib.rs` / `mod.rs`:

```rust
pub mod config;
pub mod rules;
pub mod report;

pub use config::Config;
pub use rules::RuleSet;
```

Not allowed:

```rust
pub fn analyze_repo(...) -> Result<Report> {
    ...
}
```

Instead:

```
src/
  lib.rs
  analysis/
    mod.rs
    analyze.rs
    plan.rs
```

```rust
// analysis/mod.rs
mod analyze;
mod plan;

pub use analyze::analyze_repo;
```

```rust
// analysis/analyze.rs
pub fn analyze_repo(...) -> Result<Report> {
    ...
}
```

**Rule:** `lib.rs` and `mod.rs` define the public surface. Named files contain behavior.

---

### 14. `build.rs` Follows the Same Rules

Treat `build.rs` as a tiny compile-time CLI.

Bad:

```rust
// build.rs
fn main() {
    // 200 lines of file discovery, parsing, codegen, env handling...
}
```

Better:

```
build.rs
build_support/
  mod.rs
  env.rs
  discover.rs
  generate.rs
  rerun.rs
  run.rs
```

```rust
// build.rs
mod build_support;

fn main() {
    build_support::run();
}
```

```rust
// build_support/mod.rs
mod discover;
mod env;
mod generate;
mod rerun;
mod run;

pub use run::run;
```

```rust
// build_support/run.rs
pub fn run() {
    rerun::emit_rerun_directives();
    let inputs = discover::find_inputs();
    generate::write_generated_code(inputs);
}
```

For build scripts:

| Concern | Module |
| --- | --- |
| `cargo:rerun-if-changed` | `rerun.rs` |
| Env vars | `env.rs` |
| Input discovery | `discover.rs` |
| Code generation | `generate.rs` |
| Output paths | `paths.rs` |
| Error handling | `error.rs` |

**Rule:** Build-time code deserves the same maintainability as runtime code.

---

### 15. Macro Guidance

Use macros to remove repetition, not to hide design.

**Good macro use:** repetitive trait impls, repetitive test cases, boilerplate enum conversions, static rule declarations, DSL-like configuration, generated table rows, repeated parser cases.

**Bad macro use:** hiding complex control flow, avoiding proper functions, encoding business logic in token soup, making errors harder to understand, replacing simple generics or helper functions.

#### Macro Types and Best Uses

| Macro Type | Use For | Avoid For |
| --- | --- | --- |
| `macro_rules!` | Small repetitive patterns | Complex parsing |
| Derive proc macro | Boilerplate trait impls | Behavior-heavy logic |
| Attribute proc macro | Declarative annotations | Hidden runtime effects |
| Function-like proc macro | DSLs/codegen | Ordinary helper logic |
| Build-time codegen | Large generated tables/types | Small hand-written code |

#### Prefer this order before writing a macro

1. A pure function
2. A generic function
3. A trait
4. A builder
5. A table-driven design
6. A small `macro_rules!`
7. A proc macro
8. Build-time codegen

**Rule:** Macros are for structural repetition, not unclear abstraction.

#### Keep macros small and isolated

```
src/
  lib.rs
  macros/
    mod.rs
    declare_rule.rs
    table_tests.rs
```

```rust
// macros/mod.rs
mod declare_rule;
mod table_tests;

pub(crate) use declare_rule::declare_rule;
pub(crate) use table_tests::table_tests;
```

A macro file should usually contain: one macro, one responsibility, documentation, an example expansion or usage, tests where practical.

#### Example: Macro for repetitive tests

```rust
macro_rules! metric_case {
    ($name:ident, $metric:expr, $value:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let result = check_metric($metric, $value);
            assert_eq!(result.is_ok(), $expected);
        }
    };
}

metric_case!(accepts_25_loc_fn, "fn_loc", 25, true);
metric_case!(rejects_26_loc_fn, "fn_loc", 26, false);
```

#### Example: Macro for rule declaration

```rust
macro_rules! declare_limit_rule {
    ($name:ident, $metric:expr, $limit:expr) => {
        pub fn $name(value: usize) -> RuleResult {
            check_limit($metric, value, $limit)
        }
    };
}

declare_limit_rule!(check_fn_loc, Metric::FunctionLoc, 25);
declare_limit_rule!(check_module_fns, Metric::ModuleFunctions, 5);
```

The macro expands structure; the function contains behavior:

```rust
pub fn check_limit(metric: Metric, value: usize, limit: usize) -> RuleResult {
    if value <= limit {
        RuleResult::Pass
    } else {
        RuleResult::Fail { metric, value, limit }
    }
}
```

#### Proc macro crate shape

If proc macros are needed, isolate them in their own crate:

```
crates/
  metric-core/
  metric-macros/
  metric-cli/
```

```toml
# crates/metric-macros/Cargo.toml
[lib]
proc-macro = true
```

Suggested layout:

```
metric-macros/
  src/
    lib.rs
    derive_rule.rs
    attr_metric.rs
    parse.rs
    emit.rs
    error.rs
```

`lib.rs` should only expose macro entry points:

```rust
#[proc_macro_derive(Rule)]
pub fn derive_rule(input: TokenStream) -> TokenStream {
    derive_rule::expand(input)
}
```

Actual behavior belongs in named files.

---

### 16. Final Agent Directive

Refactor and write Rust code according to these gates and layout rules:

**Metrics**
- max 25 LOC per function
- max 5 production functions per module
- max 5 modules per crate where practical
- keep repo/component/subsystem hierarchy shallow and grouped

**Layout**
- do not put functions in `lib.rs`
- do not put functions in `mod.rs`
- `lib.rs` and `mod.rs` are facade/re-export files only
- place behavior in named module files
- move tests out of production modules when they distort readability

**Design**
- prefer pure free functions for logic
- separate parsing, validation, planning, execution, rendering, and state
- separate configuration, initialization, runtime state, and arg parsing
- use traits only at real boundaries
- use facades for subsystem entry points
- use composition over god structs
- use design patterns where they reduce coupling

**Build scripts**
- `build.rs` must remain a thin entry point
- split compile-time logic into named modules
- apply the same LOC/function/module rules to build-time code

**Macros**
- use macros to remove boilerplate and repetitive structure
- do not use macros to hide complex logic
- prefer functions/generics/traits before macros
- keep macro definitions small and isolated
- `macro_rules!` is preferred for simple repetition
- proc macros require their own crate
- macro expansion should call ordinary functions where behavior lives

**Process**
- measure before refactoring
- classify the excess responsibility
- split by responsibility, not mechanically
- preserve behavior
- run tests after each coherent change
- re-measure before declaring success

**Stricter rule:** Do not add new logic to an over-limit function or module. First extract responsibilities into named pure helpers, move tests out of production modules, and re-run the metric tool.

---
