# Integrating PistaDB into an external Python project

This guide walks through the complete flow for using PistaDB from a separate
Python project, with **Linux x86_64 as the primary target**. Windows, macOS,
and Linux ARM64 paths are noted alongside.

> Goal: from a clean checkout of PistaDB to `import pistadb` and a running
> KNN search inside an unrelated Python application, in under 10 minutes.

## Architecture at a glance

```
   ┌─────────────────────────────┐         ┌────────────────────────────────┐
   │  PistaDB repo (this one)    │         │  Your Python project           │
   │                             │         │                                │
   │  src/*.c   ──cmake──▶  libs/│         │  app/                          │
   │                  linux/x86_ │         │    main.py    import pistadb   │
   │                  64/        │         │                ↓               │
   │                  libpistadb.│ ──cp──▶ │  vendor/pistadb/               │
   │                  so         │         │    libpistadb.so               │
   │  wrap/python/pistadb/       │ ─pip──▶ │  (installed into site-packages)│
   │    __init__.py (ctypes)     │         │                                │
   └─────────────────────────────┘         └────────────────────────────────┘
```

The native shared library (`libpistadb.so`) and the Python `ctypes` wrapper
(`pistadb` package) are **two independent artifacts**. The wrapper finds
the library at import time via a documented search order — see
[Step 2](#step-2--make-the-library-discoverable) for the three options.

---

## Step 1 — Build the native library

### 1a. Install build prerequisites

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install -y build-essential cmake git

# RHEL / CentOS / Rocky
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git

# Arch
sudo pacman -S --needed base-devel cmake git
```

CMake 3.15+ is required. Verify:

```bash
cmake --version
gcc --version
```

### 1b. Clone and build

```bash
git clone https://github.com/<your-fork>/PistaDB.git
cd PistaDB

bash scripts/linux/build.sh Release
```

The script auto-detects your CPU architecture via `uname -m` and writes the
output to:

| Host arch | Output path                            |
|-----------|----------------------------------------|
| x86_64    | `libs/linux/x86_64/libpistadb.so`      |
| aarch64   | `libs/linux/aarch64/libpistadb.so`     |

Confirm:

```bash
file libs/linux/$(uname -m)/libpistadb.so
ldd  libs/linux/$(uname -m)/libpistadb.so
```

`ldd` should report only standard system libraries (`libc`, `libpthread`,
`libm`). PistaDB has zero third-party runtime dependencies.

### 1c. Other platforms

```bash
# macOS  (auto-detects arm64 vs x86_64)
bash scripts/macos/build.sh Release

# Windows — run from a "Developer Command Prompt for VS"
scripts\windows\build.bat Release
```

Outputs land under `libs/macos/<arch>/libpistadb.dylib` and
`libs\windows\x64\pistadb.dll` respectively.

---

## Step 2 — Make the library discoverable

The Python wrapper resolves the native library at first import via
`pistadb._find_lib()`. Resolution order (first hit wins):

1. `PISTADB_LIB_PATH` — absolute path to a single library file
2. `PISTADB_LIB_DIR` — directory containing the library
3. The Python package directory itself (vendored install)
4. `<repo>/libs/<os>/<arch>/` (output of the build scripts)
5. `<repo>/build/`, `<repo>/build/Release`, etc.
6. `/usr/local/lib`, `/usr/lib`

Pick **one** of the three options below.

### Option A — Vendor the .so into your project (recommended)

Copy or symlink the built library into a stable location inside your
application repo, then point at it via an env var. This keeps your
deployment bundle self-contained and decouples your release from the
PistaDB checkout.

```bash
cd /path/to/your-app
mkdir -p vendor/pistadb
cp /path/to/PistaDB/libs/linux/x86_64/libpistadb.so vendor/pistadb/

# Make discovery automatic for every shell that activates your venv:
echo 'export PISTADB_LIB_DIR="$(pwd)/vendor/pistadb"' >> .envrc
# or, in a systemd unit / Dockerfile / k8s manifest:
ENV PISTADB_LIB_DIR=/app/vendor/pistadb
```

### Option B — Single absolute path via `PISTADB_LIB_PATH`

Useful when the `.so` lives outside any conventional layout
(e.g. `/opt/native-libs/libpistadb.so`).

```bash
export PISTADB_LIB_PATH=/opt/native-libs/libpistadb.so
```

This bypasses the directory search entirely.

### Option C — In-tree dev workflow

If your project lives next to a PistaDB checkout and you want hot-reload of
local C changes, no env var is needed: build with
`bash scripts/linux/build.sh` and the wrapper will find
`<PistaDB>/libs/linux/<arch>/libpistadb.so` automatically (search step 4).

---

## Step 3 — Install the Python wrapper

The wrapper is a pure-Python `ctypes` package; it does **not** bundle the
native library, so installing it is independent of Step 1.

### Option 1 — pip install from the PistaDB checkout (production)

```bash
# Inside your project's venv:
pip install /path/to/PistaDB/wrap/python/
```

### Option 2 — Editable install (development)

```bash
pip install -e /path/to/PistaDB/wrap/python/
```

Changes to `wrap/python/pistadb/__init__.py` take effect on the next import.

### Option 3 — Vendor the package directly

For maximum hermeticity (no pip dependency on the PistaDB checkout):

```bash
cp -r /path/to/PistaDB/wrap/python/pistadb your-app/vendor/
# Then add 'your-app/vendor' to PYTHONPATH or sys.path.insert(0, ...)
```

---

## Step 4 — Verify the integration

Save as `verify_pistadb.py` inside your project:

```python
"""Smoke test: build a tiny HNSW index and run a query."""
import os
import tempfile
import numpy as np
from pistadb import PistaDB, Metric, Index

DIM = 64

with tempfile.NamedTemporaryFile(suffix=".pst", delete=False) as f:
    db_path = f.name

db = PistaDB(db_path, dim=DIM, metric=Metric.L2, index=Index.HNSW)
rng = np.random.default_rng(0)

vecs = rng.random((1000, DIM), dtype="float32")
for i, v in enumerate(vecs):
    db.insert(i, v, label=f"item-{i}")

query = rng.random(DIM, dtype="float32")
results = db.search(query, k=5)
print(f"Top-5 for random query (dim={DIM}, n=1000):")
for r in results:
    print(f"  id={r.id:4d}  dist={r.distance:.4f}  label={r.label}")

db.save()
db.close()
os.unlink(db_path)
print("OK")
```

Run it:

```bash
python verify_pistadb.py
```

Expected output: 5 ranked rows and a final `OK`. Any other behaviour is
covered in [Troubleshooting](#troubleshooting).

---

## Step 5 — Production deployment

### 5a. Docker

```dockerfile
# ── builder stage: compile libpistadb.so on the same glibc as the runtime
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/<your-fork>/PistaDB.git /pistadb
RUN cd /pistadb && bash scripts/linux/build.sh Release
RUN pip wheel --no-deps -w /wheels /pistadb/wrap/python/

# ── runtime stage
FROM python:3.11-slim
COPY --from=builder /pistadb/libs/linux/x86_64/libpistadb.so /app/lib/
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install /tmp/*.whl numpy && rm /tmp/*.whl
ENV PISTADB_LIB_DIR=/app/lib
WORKDIR /app
COPY app/ /app/
CMD ["python", "main.py"]
```

Key points:
- Build the `.so` against the **same glibc** as the runtime image
  (`python:3.11-slim` is currently glibc 2.36 / Debian 12). Mixing a
  newer-glibc build with an older runtime causes `GLIBC_2.XX not found`.
- Copy only the binary you need (`libs/linux/x86_64/libpistadb.so`),
  not the whole `libs/` tree.
- Set `PISTADB_LIB_DIR` once via `ENV`; every Python process in the
  container picks it up automatically.

### 5b. Recommended target layout

```
your-app/
├── app/
│   └── main.py
├── vendor/
│   └── pistadb/
│       └── libpistadb.so      # ← copied from PistaDB/libs/linux/x86_64/
├── requirements.txt           # includes: pistadb @ file:///path/to/PistaDB/wrap/python
└── .env                       # PISTADB_LIB_DIR=./vendor/pistadb
```

### 5c. Bundling within a wheel

For a self-contained wheel, copy the `.so` into the package before
building:

```bash
cp libs/linux/x86_64/libpistadb.so wrap/python/pistadb/
pip wheel --no-deps -w dist/ wrap/python/
```

The wrapper's search step 3 (`pkg_dir`) will find it. Note that such
wheels are platform-specific and not suitable for PyPI without
`auditwheel`.

---

## Troubleshooting

### `OSError: PistaDB shared library not found`

The wrapper printed the full search path list with the error. Verify:

```bash
ls -l "$PISTADB_LIB_DIR/libpistadb.so"     # must exist and be readable
echo "$PISTADB_LIB_PATH"                   # should be unset or a real file
```

### `OSError: ... GLIBC_2.XX not found`

The `.so` was built against a newer glibc than the runtime has. Rebuild
inside a container that matches your deployment target — see the Docker
recipe above. CentOS 7 and AlmaLinux 8 are common older targets.

### `Illegal instruction (core dumped)`

Your CPU does not support AVX2 but the binary was built with AVX2 kernels.
Rebuild on (or for) the target CPU — PistaDB's CMake autodetects the host's
SIMD capabilities at configure time, so building on the deployment machine
solves it.

To force a scalar-only build on a host that *does* have AVX2 (e.g. when
producing a portable binary), edit `CMakeLists.txt` to skip the
`set(PISTADB_SIMD_SOURCES ...)` block, or build with a cross-compiler that
targets a baseline CPU.

### macOS: `dyld: Library not loaded: @rpath/libpistadb.dylib`

Set an absolute install name on the dylib:

```bash
install_name_tool -id "$(pwd)/libs/macos/$(uname -m)/libpistadb.dylib" \
                  libs/macos/$(uname -m)/libpistadb.dylib
```

Or simply use `PISTADB_LIB_PATH` with the absolute path.

### Windows: `OSError: [WinError 126] The specified module could not be found`

The `.dll` was found but failed to load — usually because `vcruntime140.dll`
or another MSVC runtime is missing. Install the
[Visual C++ Redistributable for Visual Studio 2015-2022](https://aka.ms/vs/17/release/vc_redist.x64.exe)
on the target machine.

---

## Appendix

### Environment variables, summarised

| Variable             | Type            | Behaviour                                       |
|----------------------|-----------------|-------------------------------------------------|
| `PISTADB_LIB_PATH`   | absolute file   | Highest priority; bypasses directory search.    |
| `PISTADB_LIB_DIR`    | directory       | Prepended to the search path.                   |
| (none)               | —               | Falls back to the in-tree `libs/<os>/<arch>/` and `build/` paths. |

### How to confirm SIMD acceleration is enabled

```bash
# Linux: look for AVX2 instructions in the .so
objdump -d libs/linux/x86_64/libpistadb.so | grep -m 1 vfmadd
# (any output = AVX2/FMA kernels are linked in)

# Or check at CMake-configure time — it prints one of:
#   SIMD : AVX2+FMA
#   SIMD : NEON (AArch64 built-in)
#   SIMD : scalar only
```

### Re-running the test suite against a libs/ build

From inside the PistaDB checkout:

```bash
# Clear any stale build/ tree to prove libs/ resolution is wired up:
rm -rf build/

PYTHONPATH=wrap/python pytest tests/ -q
```

All 109 tests should pass without setting `PISTADB_LIB_DIR`.
