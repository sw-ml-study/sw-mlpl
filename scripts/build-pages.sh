#!/usr/bin/env bash
set -euo pipefail

# Build pages/ for GitHub Pages deployment.
# Run this before committing pages/ changes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_DIR/apps/mlpl-web"

echo "=== Building pages/ ==="
cd "$WEB_DIR"
mkdir -p "$PROJECT_DIR/pages"
touch "$PROJECT_DIR/pages/.nojekyll"
trunk build --release --public-url /sw-mlpl/
rsync -a --delete --exclude='.nojekyll' "$PROJECT_DIR/dist/" "$PROJECT_DIR/pages/"

echo "=== Done ==="
echo "Pages built in: $PROJECT_DIR/pages/"
echo "To deploy: git add pages/ && git commit && git push"
