# scripts/

Per-OS build scripts for the PistaDB native shared library. Each script
configures CMake, builds the project, and copies the resulting binary into
`libs/<os>/<arch>/` so consumers (e.g. the Python wrapper) can locate it
without setting any environment variable.

| Host        | Run from repo root                                | Output                                  |
|-------------|---------------------------------------------------|-----------------------------------------|
| Windows x64 | `scripts\windows\build.bat [Release\|Debug]`      | `libs\windows\x64\pistadb.dll`          |
| Linux x86_64| `bash scripts/linux/build.sh [Release\|Debug]`    | `libs/linux/x86_64/libpistadb.so`       |
| Linux arm64 | `bash scripts/linux/build.sh [Release\|Debug]`    | `libs/linux/aarch64/libpistadb.so`      |
| macOS       | `bash scripts/macos/build.sh [Release\|Debug]`    | `libs/macos/<arch>/libpistadb.dylib`    |

The Linux and macOS scripts auto-detect the host architecture via `uname -m`
and place artifacts under the matching `libs/<os>/<arch>/` subdirectory. No
flag is required.

The top-level `build.bat` and `build.sh` at the repo root remain as thin
forwarders to these scripts for backward compatibility — feel free to use
either entry point.

For end-to-end instructions on consuming the built library from an external
Python project, see [`../INTEGRATION.md`](../INTEGRATION.md).
