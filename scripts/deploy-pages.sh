#!/usr/bin/env bash
# Deploy the committed ./pages/ directory to the live site by
# publishing it to the gh-pages branch (GitHub Pages build_type
# "legacy": GitHub's internal pages service publishes the branch --
# no Actions runner involved).
#
# NO FORCE-PUSH: this builds a new commit ON TOP of the CURRENT
# remote gh-pages tip (via a detached worktree at origin/gh-pages)
# and pushes it as an ordinary fast-forward. If gh-pages moved
# underneath us the push is rejected rather than clobbering it --
# which is the point. gh-pages is still a generated artifact branch
# (never hand-edited), so each deploy is just one more commit that
# replaces the published tree.
#
# Usage: ./scripts/build-pages.sh && git add pages/ && git commit
#        && ./scripts/deploy-pages.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ROOT=$(pwd)

git fetch origin gh-pages
WT=$(mktemp -d)
# Detached worktree at the current remote gh-pages tip -- no local
# gh-pages branch to collide with a previous deploy's worktree.
git worktree add --detach "$WT" origin/gh-pages
(
  cd "$WT"
  # Swap in the freshly built pages/: drop the old tracked tree,
  # copy the new one (dotfiles included). The worktree's own .git
  # link is untracked, so `git rm` leaves it alone.
  git rm -rqf . >/dev/null 2>&1 || true
  cp -a "$ROOT/pages/." .
  git add -A
  if git diff --cached --quiet; then
    echo "gh-pages already current; nothing to deploy."
  else
    git commit -q -m "deploy pages @ $(git -C "$ROOT" rev-parse --short HEAD)"
    # Parent IS the current remote tip, so this is a fast-forward.
    git push origin HEAD:gh-pages
    echo "pages/ deployed to gh-pages (fast-forward); GitHub publishes within ~1 minute."
  fi
)
git worktree remove --force "$WT"
# NOTE: no explicit build request here -- the branch push triggers
# the pages build by itself, and issuing a second request races it
# (the losing twin errors and reddens the README badge; 2026-08-06).
