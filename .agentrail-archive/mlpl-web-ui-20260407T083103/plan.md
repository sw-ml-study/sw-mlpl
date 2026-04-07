# MLPL Web UI Saga

## Quality Requirements (apply to EVERY step)

Every step MUST:
1. Follow TDD: write failing tests FIRST, then implement, then refactor
2. Pass all quality gates before committing:
   - cargo test (ALL tests pass)
   - cargo clippy --all-targets --all-features -- -D warnings (ZERO warnings)
   - cargo fmt --all (formatted)
   - markdown-checker -f "**/*.md" (if docs changed)
   - sw-checklist (project standards)
3. Update relevant docs if behavior changed
4. Use /mw-cp for checkpoint process (checks, detailed commit, push)
5. Push immediately after commit

## Goal

Build a browser-based MLPL REPL deployed to GitHub Pages.
Users visit the page, type MLPL expressions, and see results
instantly -- no install required. Includes a step-by-step
tutorial that teaches the language from basics to ML training.

Deployed at: https://sw-ml-study.github.io/sw-mlpl/

## What already exists

- mlpl-array: DenseArray with reshape, transpose, element-wise ops,
  scalar broadcasting, axis reductions, dot, matmul
- mlpl-runtime: 23 built-in functions including ML primitives
- mlpl-eval: AST-walking evaluator with environment and tracing
- mlpl-parser: full v1 syntax (literals, arrays, arithmetic,
  function calls, assignment, unary negation, repeat loops)
- mlpl-repl: CLI REPL with :help, :trace, :clear, -f flag
- mlpl-wasm: placeholder (empty lib.rs, deps on eval/trace/viz)
- mlpl-web: placeholder (stub main.rs, no deps)
- No Yew, Trunk, or GitHub Actions infrastructure exists

## Reference implementation

../../sw-embed/web-sw-cor24-apl uses:
- Yew 0.21 CSR + Trunk for WASM build
- Catppuccin Mocha theme
- pages/ directory with committed build artifacts
- .github/workflows/pages.yml deploying pages/ to GitHub Pages
- scripts/build-pages.sh running trunk build --release

Follow this pattern closely.

## Phases

### Phase 1: WASM bindings and build infrastructure
- Wire mlpl-wasm with wasm-bindgen exports
- Set up Trunk, index.html, build scripts

### Phase 2: Browser REPL
- Yew app with input/output panels
- REPL state management (environment persistence)
- Demo selector with preloaded examples

### Phase 3: Deployment
- GitHub Actions workflow
- Build, commit pages/, verify deployment

### Phase 4: Tutorial
- Interactive step-by-step tutorial built into the web UI
- Progression from scalars to ML training

## Success criteria
- Live demo at GitHub Pages URL
- Browser REPL evaluates all 23 built-in functions
- Demo selector loads example scripts
- Tutorial walks users from "1 + 2" to logistic regression
- All tests pass, all quality gates green
