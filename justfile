default:
    @just --list

check:
    cargo check

test:
    cargo test

fmt:
    cargo fmt --all

clippy:
    cargo clippy --workspace --all-targets -- -D warnings
