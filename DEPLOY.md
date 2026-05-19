# Deploy PistaDB into another project — the short version

> The detailed reference is `INTEGRATION.md`. **This file is the quick path.**
> You only ever copy **two things** and set **one environment variable**.

PistaDB = a native library (`.dll`/`.so`/`.dylib`) **+** a pure-Python wrapper
(`pistadb/`). They are independent: the wrapper finds the library at import
time via the `PISTADB_LIB_DIR` environment variable.

---

## 1. Get the native library

| Target  | Build command (run once, in the PistaDB repo)            | Library produced                         |
|---------|----------------------------------------------------------|------------------------------------------|
| Windows | `build_vs.bat`  (Developer Command Prompt)               | `libs\windows\x64\pistadb.dll`           |
| Linux   | `bash build.sh Release`                                  | `libs/linux/x86_64/libpistadb.so`        |

The library has **zero third-party runtime dependencies** (only libc/pthread/libm).
Build the Linux `.so` on (or in a container matching) the machine you deploy to.

## 2. Copy two things into your project

```
your-project/
├─ libs/pistadb/
│   └─ pistadb.dll          # Windows   (or libpistadb.so on Linux)
└─ pistadb/                 # the whole wrap/python/pistadb/ folder, copied as-is
    ├─ __init__.py
    └─ schema.py
```

```bash
# from your project root, with <PISTA> = path to the PistaDB repo
mkdir -p libs/pistadb
cp <PISTA>/libs/linux/x86_64/libpistadb.so libs/pistadb/      # Linux
#  (Windows: copy <PISTA>\libs\windows\x64\pistadb.dll  to  libs\pistadb\)

cp -r <PISTA>/wrap/python/pistadb ./pistadb                   # the wrapper
```

`numpy` is the only Python dependency: `pip install numpy`.

## 3. Tell the wrapper where the library is

The wrapper needs to find the library folder. **Recommended: don't use a
global environment variable at all** (a persisted absolute path breaks when
the project moves). Pick one:

**(a) Set it in code — best for a deployable app.** Relative to your source,
so it travels with the project. Do this before the first `PistaDB(...)` call
(simplest: top of your entry file, before `import pistadb`):

```python
import os
from pathlib import Path
os.environ.setdefault(
    "PISTADB_LIB_DIR",
    str(Path(__file__).resolve().parent / "libs" / "pistadb"))
from pistadb import PistaDB        # import AFTER setting it
```

**(b) No env var at all.** Drop the library file directly inside the copied
`pistadb/` package folder — the wrapper also searches its own directory.

**(c) A real environment variable.** Note the plain shell forms are
**session-only** (lost when the terminal closes):

```bash
export PISTADB_LIB_DIR=/abs/path/.../libs/pistadb        # this shell only
```
```powershell
$env:PISTADB_LIB_DIR = "C:\abs\path\...\libs\pistadb"    # this PowerShell only
```

To **persist** it system-wide instead:

| Where | How (persistent) |
|---|---|
| Windows, current user | `setx PISTADB_LIB_DIR "C:\path\libs\pistadb"` — *new* terminals only |
| Windows, all users | `setx /M PISTADB_LIB_DIR "C:\path\libs\pistadb"` (admin) |
| Linux/macOS, user | append `export PISTADB_LIB_DIR=/path/libs/pistadb` to `~/.bashrc` or `~/.zshrc`, then reopen the shell |
| Linux, all users | add `PISTADB_LIB_DIR=/path/libs/pistadb` (no `export`) to `/etc/environment` |
| Docker / systemd | `ENV PISTADB_LIB_DIR=/app/libs/pistadb` / unit `Environment=` |

(`setx` doesn't affect already-open windows — reopen the terminal.)

## 4. Verify (30 seconds)

```python
# verify.py — run:  python verify.py
import numpy as np
from pistadb import PistaDB, Metric, Index

with PistaDB("test.pst", dim=64, metric=Metric.L2, index=Index.HNSW) as db:
    rng = np.random.default_rng(0)
    for i, v in enumerate(rng.random((1000, 64), dtype="float32")):
        db.insert(i, v, label=f"item-{i}")
    db.save()                                  # close() does NOT auto-save
    hits = db.search(rng.random(64, dtype="float32"), k=5)
    print([(h.id, round(h.distance, 3)) for h in hits])
print("OK")
```

Expect 5 `(id, distance)` pairs and `OK`. Done.

---

### Two failure modes (the only ones you'll likely hit)

| Symptom | Fix |
|---|---|
| `OSError: PistaDB shared library not found` | `PISTADB_LIB_DIR` is wrong/unset, or the file isn't in it. `ls "$PISTADB_LIB_DIR"`. |
| Windows `[WinError 126] ... module could not be found` | Install the [MSVC 2015-2022 x64 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) on the target. |

For Docker, glibc-mismatch, wheels, SIMD checks, etc. → `INTEGRATION.md`.
