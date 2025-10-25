set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_ROOT="$(cd "$DOCS_DIR/../ama_tlbx" && pwd)"
OUTPUT_DIR="$DOCS_DIR/pdoc"

echo "📚 Generating pdoc documentation..."
echo "   Package: $PACKAGE_ROOT/src"
echo "   Output:  $OUTPUT_DIR"

# Change to package directory
cd "$PACKAGE_ROOT"

# Clean previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Generate pdoc HTML with PYTHONPATH set
PYTHONPATH=src uv run pdoc \
    -o "$OUTPUT_DIR" \
    --no-search \
    --no-show-source \
    --docformat google \
    ama_tlbx

echo "✅ pdoc documentation generated successfully"
echo "   Files: $(find "$OUTPUT_DIR" -name '*.html' | wc -l) HTML files"
