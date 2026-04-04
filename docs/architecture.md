# Architecture

MLPL is organized as a cellular monorepo with narrow crates and matching contracts.

## Dependency flow

`core -> array/parser -> runtime -> eval -> trace -> viz/wasm/apps -> ml`

## Design rules
- narrow public APIs
- contract-first development
- traceability as a first-class concern
- upstream-only visibility by default
