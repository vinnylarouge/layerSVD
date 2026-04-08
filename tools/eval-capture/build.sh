#!/usr/bin/env bash
# Build the eval-capture tool by copying the source into the cloned llama.cpp
# tree's examples directory and re-running cmake.
set -euo pipefail

HERE="$( cd "$(dirname "$0")" && pwd )"
PROJECT_ROOT="$( cd "$HERE/../.." && pwd )"
LLAMA_DIR="$PROJECT_ROOT/tools/llama.cpp"
DEST_DIR="$LLAMA_DIR/examples/eval-capture"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "error: $LLAMA_DIR not found. Run: git clone https://github.com/ggerganov/llama.cpp $LLAMA_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp "$HERE/eval-capture.cpp" "$DEST_DIR/eval-capture.cpp"
cp "$HERE/CMakeLists.txt"   "$DEST_DIR/CMakeLists.txt"

EXAMPLES_CMAKE="$LLAMA_DIR/examples/CMakeLists.txt"
if ! grep -q "add_subdirectory(eval-capture)" "$EXAMPLES_CMAKE"; then
    printf '\n# layerSVD addition\nadd_subdirectory(eval-capture)\n' >> "$EXAMPLES_CMAKE"
    echo "patched $EXAMPLES_CMAKE"
fi

cmake --build "$LLAMA_DIR/build" -j 12 --target eval-capture
echo "built: $LLAMA_DIR/build/bin/eval-capture"
