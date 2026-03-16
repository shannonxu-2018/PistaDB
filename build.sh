#!/usr/bin/env bash
# PistaDB build script
# Usage: bash build.sh [Release|Debug]
set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="build"

echo "=== PistaDB Build ==="
echo "Build type: $BUILD_TYPE"
echo "CMake: $(cmake --version | head -1)"

cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON

cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j

echo ""
echo "=== Build complete ==="
echo "Shared library:"
find "$BUILD_DIR" -name "*.dll" -o -name "*.so" -o -name "*.dylib" 2>/dev/null | head -5

echo ""
echo "To install the Python package:"
echo "  pip install -e python/"
echo ""
echo "To run tests:"
echo "  PISTADB_LIB_DIR=build/Release pytest tests/ -v"
echo ""
echo "To run the example:"
echo "  PISTADB_LIB_DIR=build/Release python examples/example.py"
