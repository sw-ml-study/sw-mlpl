#!/usr/bin/env bash
# List every .mlpl example (recursively under examples/) with its line
# count. Usage: examples/scripts/list.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

printf '%5s  %s\n' LINES FILE
every_example | while read -r f; do
    printf '%5s  %s\n' "$(wc -l < "$f" | tr -d ' ')" "${f#"$REPO_ROOT"/}"
done
