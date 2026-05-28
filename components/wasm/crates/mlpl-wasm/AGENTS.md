# mlpl-wasm agent notes

## Purpose
Local implementation capsule for `mlpl-wasm`.

## This crate should know
- local public API
- local tests
- matching contract

## This crate should not know
- unrelated downstream crate internals

## Rule
Stay within local task scope and escalate before widening public APIs.
