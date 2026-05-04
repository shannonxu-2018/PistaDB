#!/usr/bin/env bash
# =============================================================================
# PistaDB - Linux build script (GCC / Clang)
#
# Usage:
#   bash scripts/linux/build.sh           # defaults to Release
#   bash scripts/linux/build.sh Debug
#
# Output (architecture auto-detected via uname -m):
#   libs/linux/x86_64/libpistadb.so       (on x86_64 hosts)
#   libs/linux/aarch64/libpistadb.so      (on aarch64/arm64 hosts)
#   libs/linux/<arch>/libpistadb_static.a (if the static target was built)
#
# Requirements:
#   apt:  sudo apt install -y build-essential cmake
#   yum:  sudo yum groupinstall -y "Development Tools" && sudo yum install -y cmake
# =============================================================================
set -euo pipefail

BUILD_TYPE="${1:-Release}"

# Resolve repo root: this script lives at <root>/scripts/linux/build.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

# Architecture detection. Normalise common aliases so the layout matches the
# Python wrapper's _find_lib() expectations.
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
    x86_64|amd64)  ARCH="x86_64"  ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *)
        echo "ERROR: unsupported architecture '$ARCH_RAW' for the Linux build script." >&2
        echo "       Supported: x86_64, aarch64." >&2
        echo "       To build manually: cmake -B build && cmake --build build && cp build/libpistadb.so libs/linux/$ARCH_RAW/" >&2
        exit 2
        ;;
esac

OUT_DIR="$REPO_ROOT/libs/linux/$ARCH"

echo "=== PistaDB Linux Build ==="
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

# Single-config generator: artifacts land directly in build/.
cp -v "$BUILD_DIR/libpistadb.so" "$OUT_DIR/"
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
