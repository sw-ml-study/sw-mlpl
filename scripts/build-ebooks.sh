#!/usr/bin/env bash
set -euo pipefail

# Build books/ as EPUB, PDF, and HTML.
# Requires: mdbook, mdbook-epub, mdbook-pdf

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BOOKS_DIR="$PROJECT_DIR/books"
DIST_DIR="$PROJECT_DIR/books/dist"

echo "=== Building MLPL Ebooks ==="

# Check for mdbook
if ! command -v mdbook &> /dev/null; then
    echo "Error: mdbook is not installed. Install with 'cargo install mdbook'"
    exit 1
fi

# Check for plugins
if ! command -v mdbook-epub &> /dev/null; then
    echo "Warning: mdbook-epub is not installed. EPUB generation may fail."
    echo "Install with 'cargo install mdbook-epub'"
fi

if ! command -v mdbook-pdf &> /dev/null; then
    echo "Warning: mdbook-pdf is not installed. PDF generation may fail."
    echo "Install with 'cargo install mdbook-pdf'"
fi

mkdir -p "$DIST_DIR"

for book in lang-reference user-guide comprehensive-guide; do
    echo "--- Building $book ---"
    cd "$BOOKS_DIR/$book"
    mdbook build
    
    # Organize outputs
    mkdir -p "$DIST_DIR/$book"
    [ -d "book/html" ] && rsync -a --delete "book/html/" "$DIST_DIR/$book/html/"
    [ -f "book/epub/MLPL $book.epub" ] && cp "book/epub/MLPL $book.epub" "$DIST_DIR/$book.epub"
    [ -f "book/pdf/output.pdf" ] && cp "book/pdf/output.pdf" "$DIST_DIR/$book.pdf"
done

echo "=== Done ==="
echo "Ebooks generated in: $DIST_DIR"
