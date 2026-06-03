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
WEB_DIR="$PROJECT_DIR/components/web/crates/mlpl-web"
PAGES_DIST="$PROJECT_DIR/dist-pages"

echo "=== Building pages/ ==="
cd "$WEB_DIR"
mkdir -p "$PROJECT_DIR/pages"
touch "$PROJECT_DIR/pages/.nojekyll"
trunk build --release --public-url /sw-mlpl/ --dist "$PAGES_DIST"
rsync -a --delete --exclude='.nojekyll' "$PAGES_DIST/" "$PROJECT_DIR/pages/"

# Cache-bust js/stage3d.js. Unlike the trunk WASM bundle, this
# file is referenced by a stable, non-hashed URL, so browsers
# cache it and keep serving stale JS across deploys. Stamp the
# <script src> with a content hash so each change forces a
# refetch (no manual hard-refresh needed).
STAGE_JS="$PROJECT_DIR/pages/js/stage3d.js"
INDEX="$PROJECT_DIR/pages/index.html"
if [ -f "$STAGE_JS" ] && [ -f "$INDEX" ]; then
    # Portable content hash: md5sum (Linux) or md5 -q (macOS).
    if command -v md5sum >/dev/null 2>&1; then
        VER=$(md5sum "$STAGE_JS" | cut -c1-12)
    else
        VER=$(md5 -q "$STAGE_JS" 2>/dev/null | cut -c1-12)
    fi
    [ -z "$VER" ] && VER=$(date +%s)
    # `sed -i.bak` is portable across GNU and BSD sed (bare `-i ''`
    # is BSD-only and errors on Linux).
    sed -i.bak "s|js/stage3d\.js[^\"\']*|js/stage3d.js?v=$VER|g" "$INDEX"
    rm -f "$INDEX.bak"
    echo "Stamped stage3d.js cache-bust: ?v=$VER"
fi

# Bundle the literate-programming HTML outputs (companions to the
# connect-only demos) into pages/literate/. The rsync --delete above
# only mirrors the trunk dist, so copy these after it. Demos link to
# them via the registry's LITERATE_DOCS map.
LITERATE_SRC="$PROJECT_DIR/examples/literate"
if [ -d "$LITERATE_SRC" ]; then
    mkdir -p "$PROJECT_DIR/pages/literate"
    cp "$LITERATE_SRC"/*.html "$PROJECT_DIR/pages/literate/" 2>/dev/null || true
    echo "Bundled literate HTML: $(ls "$PROJECT_DIR/pages/literate" | tr '\n' ' ')"
fi

echo "=== Done ==="
echo "Pages built in: $PROJECT_DIR/pages/"
echo "To deploy: git add pages/ && git commit && git push"
