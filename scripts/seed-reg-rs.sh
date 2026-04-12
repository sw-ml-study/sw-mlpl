#!/usr/bin/env bash
# Seed reg-rs regression tests for all on-disk and web-extracted MLPL
# demos. Run once to create baselines, then `reg-rs run` on every
# commit to catch regressions.
#
# Usage: scripts/seed-reg-rs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PREPROCESS="bash scripts/normalize-mlpl-output.sh"
CMD_PREFIX="cargo run --release --quiet -p mlpl-repl -- -f"

echo "=== Building mlpl-repl (release) ==="
cargo build --release --quiet -p mlpl-repl

echo ""
echo "=== Seeding on-disk demos (demos/*.mlpl) ==="
for demo in demos/*.mlpl; do
    base="$(basename "$demo" .mlpl)"
    test_name="mlpl-demo-${base}"

    if [[ -f "work/reg-rs/${test_name}.rgt" ]]; then
        echo "  skip $test_name (already exists)"
        continue
    fi

    echo "  creating $test_name ..."
    reg-rs create \
        -t "$test_name" \
        -c "$CMD_PREFIX $demo" \
        -P "$PREPROCESS" \
        --timeout 120 \
        --desc "On-disk demo: $demo"
done

echo ""
echo "=== Extracting web demos ==="
bash scripts/gen-web-demos.sh

echo ""
echo "=== Seeding web-extracted demos ==="
for demo in work/reg-rs/web-demos/*.mlpl; do
    base="$(basename "$demo" .mlpl)"
    test_name="mlpl-web-${base}"

    if [[ -f "work/reg-rs/${test_name}.rgt" ]]; then
        echo "  skip $test_name (already exists)"
        continue
    fi

    echo "  creating $test_name ..."
    reg-rs create \
        -t "$test_name" \
        -c "$CMD_PREFIX $demo" \
        -P "$PREPROCESS" \
        --timeout 120 \
        --desc "Web demo extracted from demos.rs: $base"
done

echo ""
echo "=== Verifying all tests pass ==="
reg-rs run -q && echo "ALL PASS" || echo "SOME FAILURES"

echo ""
echo "=== Summary ==="
reg-rs list
