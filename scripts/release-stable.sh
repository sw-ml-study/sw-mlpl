#!/usr/bin/env bash
set -euo pipefail

# Cut a STABLE public release of the live demo.
#
# Builds the site at a ROOT base-path (--public-url /) -- unlike the
# rolling dev deploy, which uses /sw-mlpl/ for the project-pages subpath
# -- and mirrors it into ../mlpl-live, the checkout for the custom-domain
# GitHub Pages repo served at https://mlpl.softwarewrighter.com/.
#
# This is the deliberate "ship a stable version" action: run it ONLY when
# you want the public demo to move. Pushing to sw-mlpl (which auto-deploys
# the /sw-mlpl/ dev demo via build-pages.sh + deploy-pages.sh) does NOT
# touch the stable release until you run this and push ../mlpl-live.
#
# First run creates ../mlpl-live as a fresh git repo and commits; then:
#   1. create an EMPTY repo sw-ml-study/mlpl-live on GitHub
#   2. git -C ../mlpl-live remote add origin git@github.com:sw-ml-study/mlpl-live.git
#   3. git -C ../mlpl-live push -u origin main
#   4. GitHub repo Settings -> Pages: Source = main / root; confirm the
#      custom domain mlpl.softwarewrighter.com (read from the CNAME file);
#      enable "Enforce HTTPS" once the cert provisions.
# Later runs just rebuild + commit; re-push to republish.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_DIR/components/web/crates/mlpl-web"
REL_DIST="$PROJECT_DIR/dist-release"
MIRROR="$PROJECT_DIR/../mlpl-live"
DOMAIN="mlpl.softwarewrighter.com"

echo "=== Building STABLE release (root base-path) ==="
cd "$WEB_DIR"
"$SCRIPT_DIR/serial.sh" trunk build --release --public-url / --dist "$REL_DIST"

mkdir -p "$MIRROR"
# Mirror the built site; preserve the repo's own .git and the files this
# script (re)writes below across rebuilds.
rsync -a --delete \
    --exclude='.git/' --exclude='CNAME' --exclude='.nojekyll' --exclude='literate/' \
    "$REL_DIST/" "$MIRROR/"

touch "$MIRROR/.nojekyll"
echo "$DOMAIN" >"$MIRROR/CNAME"

# Literate companion pages: copy the repo's committed set and rewrite the
# dev subpath (/sw-mlpl/) to root so their asset refs resolve at the domain
# root (harmless no-op if a page has no absolute dev-base refs).
if [ -d "$PROJECT_DIR/pages/literate" ]; then
    mkdir -p "$MIRROR/literate"
    cp "$PROJECT_DIR/pages/literate/"*.html "$MIRROR/literate/" 2>/dev/null || true
    for f in "$MIRROR/literate/"*.html; do
        [ -f "$f" ] || continue
        sed -i.bak 's|/sw-mlpl/|/|g' "$f" && rm -f "$f.bak"
    done
fi

# stage3d.js cache-bust: stable, non-hashed URL, so stamp a content hash
# (same technique as build-pages.sh).
STAGE_JS="$MIRROR/js/stage3d.js"
INDEX="$MIRROR/index.html"
if [ -f "$STAGE_JS" ] && [ -f "$INDEX" ]; then
    if command -v md5sum >/dev/null 2>&1; then
        VER=$(md5sum "$STAGE_JS" | cut -c1-12)
    else
        VER=$(md5 -q "$STAGE_JS" 2>/dev/null | cut -c1-12)
    fi
    [ -z "$VER" ] && VER=$(date +%s)
    sed -i.bak "s|js/stage3d\.js[^\"\']*|js/stage3d.js?v=$VER|g" "$INDEX"
    rm -f "$INDEX.bak"
fi

# Build provenance: the stable channel's own build-info.json + page meta.
COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUNDLE=$(ls "$MIRROR"/mlpl-web-*_bg.wasm | head -1 | sed 's/.*\(mlpl-web-[0-9a-f]*\)_bg.wasm/\1/')
cat >"$MIRROR/build-info.json" <<INFO
{"commit":"$COMMIT","built_at":"$BUILT_AT","bundle":"$BUNDLE","channel":"stable"}
INFO
perl -pi -e "s|<head>|<head><meta name=\"mlpl-build\" content=\"$COMMIT $BUILT_AT stable\">|" "$INDEX"

# Git: init on first run, then commit this release.
if [ ! -d "$MIRROR/.git" ]; then
    git -C "$MIRROR" init -q -b main
    echo "Initialized $MIRROR as a fresh git repo (branch main)."
fi
git -C "$MIRROR" add -A
git -C "$MIRROR" commit -q -m "release: sw-mlpl $COMMIT (stable @ $BUILT_AT)" || echo "(nothing new to commit)"

echo "=== Stable release built in $MIRROR (from sw-mlpl $COMMIT) ==="
if ! git -C "$MIRROR" remote get-url origin >/dev/null 2>&1; then
    cat <<NEXT
Next (first time only):
  1. Create an EMPTY repo sw-ml-study/mlpl-live on GitHub (no README/license).
  2. git -C $MIRROR remote add origin git@github.com:sw-ml-study/mlpl-live.git
  3. git -C $MIRROR push -u origin main
  4. Repo Settings -> Pages: Source = main / root; custom domain
     $DOMAIN (from CNAME); enable Enforce HTTPS after the cert provisions.
NEXT
else
    echo "To publish: git -C $MIRROR push"
fi
