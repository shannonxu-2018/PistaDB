# libs/

Output directory for the PistaDB native shared library, populated by the
per-OS build scripts under [`../scripts/`](../scripts/). The directory
structure is committed (this README and `.gitkeep` files), but the built
binaries themselves are git-ignored — every developer or deployment target
produces their own.

## Layout

```
libs/
├── windows/
│   └── x64/
│       ├── pistadb.dll
│       ├── pistadb.lib              # DLL import library
│       └── pistadb_static.lib
├── linux/
│   ├── x86_64/
│   │   ├── libpistadb.so
│   │   └── libpistadb_static.a
│   └── aarch64/
│       ├── libpistadb.so
│       └── libpistadb_static.a
└── macos/
    ├── arm64/
    │   ├── libpistadb.dylib
    │   └── libpistadb_static.a
    └── x86_64/
        ├── libpistadb.dylib
        └── libpistadb_static.a
```

## Populating

```bash
# Linux  (auto-detects x86_64 vs aarch64 via uname -m)
bash scripts/linux/build.sh

# macOS  (auto-detects arm64 vs x86_64)
bash scripts/macos/build.sh

# Windows
scripts\windows\build.bat
```

## Discovery

The Python wrapper (`wrap/python/pistadb`) automatically searches
`libs/<os>/<arch>/` for the matching binary at import time. No environment
variable is required as long as the wrapper is imported from a checkout that
contains a populated `libs/` tree at the same relative location.

For external Python projects (where `libs/` is somewhere else on disk), see
[`../INTEGRATION.md`](../INTEGRATION.md) for the full set of options
(`PISTADB_LIB_DIR`, `PISTADB_LIB_PATH`, vendoring).
