#!/usr/bin/env bash
# build.sh — Build PistaDB WebAssembly binding with Emscripten
#
# Requirements:
#   • Emscripten SDK installed and activated in the current shell:
#       source /path/to/emsdk/emsdk_env.sh
#   • CMake ≥ 3.15
#
# Usage:
#   cd wasm
#   bash build.sh [Release|Debug]   # default: Release
#
# Output:
#   wasm/build/pistadb.js    — JavaScript module factory
#   wasm/build/pistadb.wasm  — WebAssembly binary

set -euo pipefail

BUILD_TYPE="${1:-Release}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "► Checking Emscripten..."
if ! command -v emcmake &>/dev/null; then
    echo "  ERROR: emcmake not found."
    echo "  Activate the Emscripten SDK first:"
    echo "    source /path/to/emsdk/emsdk_env.sh"
    exit 1
fi
emcc --version | head -1

echo "► Configuring (${BUILD_TYPE})..."
mkdir -p "$BUILD_DIR"
emcmake cmake \
    -S "$SCRIPT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "► Building..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -- -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo ""
echo "✓ Build complete:"
echo "  $BUILD_DIR/pistadb.js"
echo "  $BUILD_DIR/pistadb.wasm"
echo ""
echo "To use in a browser, serve both files from the same HTTP origin."
echo "To use in Node.js:"
echo "  const PistaDB = require('./pistadb.js');"
echo "  const M = await PistaDB();"
