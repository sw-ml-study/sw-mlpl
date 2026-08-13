#!/usr/bin/env bash
# Deploy the built ./pages/ to the live site (GitHub Pages, build_type
# "legacy": GitHub publishes the gh-pages branch directly, no Actions
# runner).
#
# gh-pages holds the SITE at its root, while main holds it under
# pages/, so gh-pages gets its own checkout -- exactly like the wiki
# (../sw-mlpl.wiki). Deploying is then just plain git: mirror the built
# files in, then add / commit / push. No subtree, no force, no
# per-deploy worktree churn.
#
# One-time setup (already done; recreate only if ../sw-mlpl.pages is
# missing):
#   git fetch origin gh-pages
#   git worktree add -b gh-pages ../sw-mlpl.pages origin/gh-pages
#
# Usage: ./scripts/build-pages.sh && git add pages/ && git commit
#        && ./scripts/deploy-pages.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DEPLOY="../sw-mlpl.pages"

if [ ! -d "$DEPLOY/.git" ] && [ ! -f "$DEPLOY/.git" ]; then
  echo "error: $DEPLOY is not a gh-pages checkout. One-time setup:" >&2
  echo "  git fetch origin gh-pages" >&2
  echo "  git worktree add -b gh-pages ../sw-mlpl.pages origin/gh-pages" >&2
  exit 1
fi

# Mirror the freshly built site into the gh-pages checkout (a plain
# file copy that also drops stale hashed bundles; not a git command).
rsync -a --delete --exclude='.git' pages/ "$DEPLOY/"

git -C "$DEPLOY" add -A
git -C "$DEPLOY" commit -m "deploy pages @ $(git rev-parse --short HEAD)" \
  || { echo "gh-pages already current; nothing to deploy."; exit 0; }
git -C "$DEPLOY" push
echo "deployed to gh-pages; GitHub publishes within ~1 minute."
