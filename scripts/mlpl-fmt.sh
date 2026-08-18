#!/usr/bin/env bash
# mlpl-fmt -- format .mlpl source by MLPL major-mode indentation,
# using batch Emacs. The MLPL analogue of `cargo fmt`: it reindents
# every line to its brace/bracket/paren depth (MLPL's if/else/while/
# repeat/def blocks are brace-delimited) and strips trailing
# whitespace. Because mlpl-mode's indenter is absolute and idempotent
# (docs: elisp/mlpl-mode.el `mlpl--calculate-indent'), running this on
# already-formatted code is a no-op.
#
# Usage:
#   scripts/mlpl-fmt.sh [--check] [FILE|DIR ...]
#
#   (no paths)   format every tracked *.mlpl file in the repo
#   FILE ...     format the named .mlpl files
#   DIR ...      format every *.mlpl under the directory (recursive)
#   --check      do not write; exit non-zero if any file WOULD change
#                (prints the offenders) -- for CI / pre-commit, like
#                `cargo fmt -- --check`
#
# Emacs need not be on PATH: set $EMACS, or the script finds a macOS
# Emacs.app / Homebrew install automatically.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode_el="$repo_root/elisp/mlpl-mode.el"

find_emacs() {
  if [[ -n "${EMACS:-}" ]] && command -v "$EMACS" >/dev/null 2>&1; then
    echo "$EMACS"; return 0
  fi
  local cand
  for cand in emacs emacs-nox \
    /Applications/Emacs.app/Contents/MacOS/Emacs \
    /opt/homebrew/bin/emacs /usr/local/bin/emacs; do
    if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

emacs_bin="$(find_emacs || true)"
if [[ -z "$emacs_bin" ]]; then
  echo "mlpl-fmt: no Emacs found. Install Emacs or set \$EMACS to its path." >&2
  exit 127
fi
if [[ ! -f "$mode_el" ]]; then
  echo "mlpl-fmt: cannot find mode file at $mode_el" >&2
  exit 1
fi

check=0
paths=()
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "mlpl-fmt: unknown option: $arg" >&2; exit 2 ;;
    *) paths+=("$arg") ;;
  esac
done

# Collect the target file list.
files=()
if [[ ${#paths[@]} -eq 0 ]]; then
  while IFS= read -r f; do files+=("$f"); done \
    < <(cd "$repo_root" && git ls-files '*.mlpl')
  # git ls-files is repo-relative; make absolute.
  for i in "${!files[@]}"; do files[$i]="$repo_root/${files[$i]}"; done
else
  for p in "${paths[@]}"; do
    if [[ -d "$p" ]]; then
      while IFS= read -r f; do files+=("$f"); done \
        < <(find "$p" -type f -name '*.mlpl')
    elif [[ -f "$p" ]]; then
      files+=("$p")
    else
      echo "mlpl-fmt: no such file or directory: $p" >&2; exit 1
    fi
  done
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "mlpl-fmt: no .mlpl files to format."
  exit 0
fi

# Emacs batch program: format the visited file in place. Called on a
# TEMP copy so --check can diff without touching the original.
format_into() { # $1 = file to format in place
  "$emacs_bin" -Q --batch \
    --load "$mode_el" \
    --visit "$1" \
    --eval '(progn
              (mlpl-mode)
              (indent-region (point-min) (point-max))
              (delete-trailing-whitespace)
              (save-buffer))' \
    >/dev/null 2>&1
}

changed=0
tmp="$(mktemp -t mlpl-fmt.XXXXXX)"
trap 'rm -f "$tmp"' EXIT

for f in "${files[@]}"; do
  cp "$f" "$tmp"
  format_into "$tmp"
  if cmp -s "$f" "$tmp"; then
    continue
  fi
  changed=$((changed + 1))
  if [[ $check -eq 1 ]]; then
    echo "would reformat: $f"
  else
    cp "$tmp" "$f"
    echo "formatted: $f"
  fi
done

if [[ $check -eq 1 ]]; then
  if [[ $changed -gt 0 ]]; then
    echo "mlpl-fmt: $changed file(s) need formatting." >&2
    exit 1
  fi
  echo "mlpl-fmt: all ${#files[@]} file(s) already formatted."
else
  echo "mlpl-fmt: formatted $changed of ${#files[@]} file(s)."
fi
