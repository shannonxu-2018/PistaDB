#!/usr/bin/env bash
# =============================================================================
# PistaDB - macOS build script (Apple Clang)
#
# Usage:
#   bash scripts/macos/build.sh           # defaults to Release
#   bash scripts/macos/build.sh Debug
#
# Output (architecture auto-detected via uname -m):
#   libs/macos/arm64/libpistadb.dylib     (Apple Silicon)
#   libs/macos/x86_64/libpistadb.dylib    (Intel)
#
# Requirements:
#   - Xcode Command Line Tools: xcode-select --install
#   - CMake 3.15+ (brew install cmake)
#
# Note: this script builds for the host architecture only. To produce a
# universal binary, pass -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" yourself
# and copy the result into both libs/macos/arm64/ and libs/macos/x86_64/.
# =============================================================================
set -euo pipefail

BUILD_TYPE="${1:-Release}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
    arm64|aarch64) ARCH="arm64"  ;;
    x86_64|amd64)  ARCH="x86_64" ;;
    *)
        echo "ERROR: unsupported macOS architecture '$ARCH_RAW'." >&2
        echo "       Supported: arm64 (Apple Silicon), x86_64 (Intel)." >&2
        exit 2
        ;;
esac

OUT_DIR="$REPO_ROOT/libs/macos/$ARCH"

echo "=== PistaDB macOS Build ==="
echo "Repo root  : $REPO_ROOT"
echo "Build type : $BUILD_TYPE"
echo "Host arch  : $ARCH_RAW (normalised to $ARCH)"
echo "Build dir  : $BUILD_DIR"
echo "Output dir : $OUT_DIR"
echo "CMake      : $(cmake --version | head -1)"
echo

cmake -B "$BUILD_DIR" -S "$REPO_ROOT" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j

mkdir -p "$OUT_DIR"

cp -v "$BUILD_DIR/libpistadb.dylib" "$OUT_DIR/"
if [ -f "$BUILD_DIR/libpistadb_static.a" ]; then
    cp -v "$BUILD_DIR/libpistadb_static.a" "$OUT_DIR/"
fi

echo
echo "=== Build complete ==="
echo "Artifacts:"
ls -lh "$OUT_DIR"
echo
echo "To use from Python:"
echo "  export PISTADB_LIB_DIR=\"$OUT_DIR\""
echo "  python -c 'import pistadb; print(pistadb.__file__)'"
echo
echo "To run tests:"
echo "  export PISTADB_LIB_DIR=\"$OUT_DIR\""
echo "  PYTHONPATH=\"$REPO_ROOT/wrap/python\" pytest tests/ -v"
