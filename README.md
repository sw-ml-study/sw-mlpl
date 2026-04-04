# MLPL

MLPL is a Rust-first array and tensor programming language for machine learning, visualization, and experimentation.

This repository is named `sw-mlpl`, while the language and platform described in the documentation are called **MLPL**.

MLPL is inspired by APL, APL2, J, and BQN, and is designed to support:

- dense tensor and array programming
- visual debugging and execution tracing
- ML experiments in Rust
- contract-first, compartmentalized implementation by multiple coding agents

## Operating model

MLPL is intended to be developed as a **cellular monorepo**:

- each crate is a small capsule
- each implementation area has a matching `contracts/` directory
- most coding tasks should touch one crate plus one contract
- each agent should start with minimal context

See:

- `AGENTS.md`
- `COORDINATOR.md`
- `docs/repo-structure.md`
- `docs/agent-coordination.md`
- `docs/saga.md`

## Naming

- Git repository name: `sw-mlpl`
- Product, language, and platform name in docs: **MLPL**
