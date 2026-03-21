#!/bin/bash

# Render Marp slides to PDF
# Usage:
#   ./render.sh              - render all slides
#   ./render.sh <file>.md    - render one specific slide file

set -e

SLIDES_DIR="$(cd "$(dirname "$0")" && pwd)"
PDF_DIR="$SLIDES_DIR/pdfs"

# Create pdfs directory if it doesn't exist
mkdir -p "$PDF_DIR"

# Function to render a single markdown file
render_slide() {
    local md_file="$1"
    local basename=$(basename "$md_file" .md)
    local pdf_file="$PDF_DIR/${basename}.pdf"
    
    echo "Rendering: $md_file → $pdf_file"
    marp --pdf --allow-local-files "$md_file" -o "$pdf_file"
}

# If no argument provided, render all
if [ $# -eq 0 ]; then
    echo "Rendering all slides..."
    for md_file in "$SLIDES_DIR"/*.md; do
        # Skip README.md
        if [[ $(basename "$md_file") == "README.md" ]]; then
            continue
        fi
        render_slide "$md_file"
    done
    echo "✓ All slides rendered successfully!"
else
    # Render exactly one file (no pattern matching)
    target_file="$1"

    if [[ "$target_file" != *.md ]]; then
        echo "Error: Please provide the full markdown filename (e.g., 02_hidden_markov_models.md)."
        echo ""
        echo "Available slides:"
        ls -1 "$SLIDES_DIR"/*.md | grep -v README.md | xargs -n1 basename
        exit 1
    fi

    md_file="$SLIDES_DIR/$target_file"
    if [[ ! -f "$md_file" ]] || [[ "$target_file" == "README.md" ]]; then
        echo "Error: Slide file '$target_file' not found."
        echo ""
        echo "Available slides:"
        ls -1 "$SLIDES_DIR"/*.md | grep -v README.md | xargs -n1 basename
        exit 1
    fi

    render_slide "$md_file"
    echo "✓ Done!"
fi
