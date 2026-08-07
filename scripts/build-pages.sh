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
"$SCRIPT_DIR/serial.sh" trunk build --release --public-url /sw-mlpl/ --dist "$PAGES_DIST"
# Exclude literate/ from --delete: the deployed literate HTML is COMMITTED
# under pages/literate/ (examples/literate/*.html is gitignored/transient).
# Without this, --delete wipes pages/literate/ -- which is how the literate
# pages were lost on past rebuilds. The cp below re-bundles any freshly
# published examples/literate/*.html over the committed ones.
rsync -a --delete --exclude='.nojekyll' --exclude='literate/' "$PAGES_DIST/" "$PROJECT_DIR/pages/"

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

# Build provenance: a machine-readable stamp served beside the
# site (badges + the splash staleness check read it) and a meta
# tag baked into the page itself (the RUNNING page's identity).
COMMIT=$(git rev-parse --short HEAD)
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUNDLE=$(ls "$PROJECT_DIR"/pages/mlpl-web-*_bg.wasm | head -1 | sed 's/.*\(mlpl-web-[0-9a-f]*\)_bg.wasm/\1/')
LEDGER=$(cd "$PROJECT_DIR" && { sw-checklist 2>/dev/null || true; } | perl -pe 's/\e\[[0-9;]*m//g' | grep -o 'Summary: .*' | head -1 | sed 's/Summary: //')
LEDGER=${LEDGER:-unknown}
cat > "$PROJECT_DIR/pages/build-info.json" <<INFO
{"commit":"$COMMIT","built_at":"$BUILT_AT","bundle":"$BUNDLE","gates":"local pre-commit (tests + clippy + fmt + sw-checklist)","ledger":"$LEDGER"}
INFO
# Stamp the page: perl -pi keeps the file's encoding intact.
perl -pi -e "s|<head>|<head><meta name=\"mlpl-build\" content=\"$COMMIT $BUILT_AT\">|" "$PROJECT_DIR/pages/index.html"
echo "build-info: $COMMIT @ $BUILT_AT"
