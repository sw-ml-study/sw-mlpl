#!/usr/bin/env bash
# Deploy the committed ./pages/ directory to the live site by
# pushing it to the gh-pages branch (GitHub Pages build_type
# "legacy": GitHub's internal pages service publishes the
# branch -- no Actions runner involved). Force-push is correct
# here: gh-pages is a GENERATED artifact branch, never edited
# by hand and never shared for development.
#
# Usage: ./scripts/build-pages.sh && git add pages/ && git commit
#        && ./scripts/deploy-pages.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git subtree split --prefix pages -b _pages_deploy HEAD
git push -f origin _pages_deploy:gh-pages
git branch -D _pages_deploy
echo "pages/ pushed to gh-pages; GitHub publishes it within ~1 minute."
