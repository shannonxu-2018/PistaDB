#!/usr/bin/env bash
# Backward-compatible forwarder. Dispatches to the per-OS script under
#   scripts/<os>/build.sh
# which additionally copies artifacts into libs/<os>/<arch>/.
# See scripts/README.md and INTEGRATION.md for the full per-OS layout.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Linux*)   exec bash "$SCRIPT_DIR/scripts/linux/build.sh"  "$@" ;;
    Darwin*)  exec bash "$SCRIPT_DIR/scripts/macos/build.sh"  "$@" ;;
    MINGW*|MSYS*|CYGWIN*)
        # Bash on Windows — defer to the .bat for proper MSVC integration.
        exec cmd.exe //c "$SCRIPT_DIR/scripts/windows/build.bat" "$@" ;;
    *)
        echo "Unsupported host OS '$(uname -s)'." >&2
        echo "Available scripts: scripts/{windows,linux,macos}/build.{bat,sh}" >&2
        exit 2 ;;
esac
