#!/usr/bin/env bash
set -euo pipefail

# Build pages/ for GitHub Pages deployment.
# Run this before committing pages/ changes.
#
# Uses a SEPARATE dist dir (`dist-pages/`) from `trunk serve`'s
# `dist/`. Sharing the dir lets the pages build clobber serve's
# bundle hashes mid-flight, which then triggers
# "Failed to find a valid digest in the 'integrity' attribute"
# in the browser because the served HTML references one bundle
# hash and the actual file on disk has a different one.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_DIR/apps/mlpl-web"
PAGES_DIST="$PROJECT_DIR/dist-pages"

echo "=== Building pages/ ==="
cd "$WEB_DIR"
mkdir -p "$PROJECT_DIR/pages"
touch "$PROJECT_DIR/pages/.nojekyll"
trunk build --release --public-url /sw-mlpl/ --dist "$PAGES_DIST"
rsync -a --delete --exclude='.nojekyll' "$PAGES_DIST/" "$PROJECT_DIR/pages/"

echo "=== Done ==="
echo "Pages built in: $PROJECT_DIR/pages/"
echo "To deploy: git add pages/ && git commit && git push"
