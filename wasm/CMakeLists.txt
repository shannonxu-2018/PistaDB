# wasm/CMakeLists.txt — Emscripten build for PistaDB WebAssembly binding
#
# Usage:
#   cd wasm
#   emcmake cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Output: build/pistadb.js + build/pistadb.wasm
#
# Requires Emscripten SDK (emsdk) to be active in the current shell.
# See: https://emscripten.org/docs/getting_started/downloads.html

cmake_minimum_required(VERSION 3.15)
project(pistadb_wasm LANGUAGES C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Source files (same C core as the native build) ────────────────────────────
set(PISTADB_SOURCES
    ../src/distance.c
    ../src/utils.c
    ../src/storage.c
    ../src/index_linear.c
    ../src/index_hnsw.c
    ../src/index_ivf.c
    ../src/index_ivf_pq.c
    ../src/index_diskann.c
    ../src/index_lsh.c
    ../src/index_scann.c
    ../src/pistadb.c
)

# ── Build target ──────────────────────────────────────────────────────────────
# Emscripten compiles executables to .js + .wasm pairs.
add_executable(pistadb_wasm
    pistadb_wasm.cpp
    ${PISTADB_SOURCES}
)

target_include_directories(pistadb_wasm PRIVATE ../src)

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(pistadb_wasm PRIVATE
    -O3
    -ffast-math
    -fwasm-exceptions           # native WASM exceptions (Emscripten 3.x+)
)

# ── Output name ───────────────────────────────────────────────────────────────
# Emscripten appends .js automatically; the .wasm file gets the same stem.
set_target_properties(pistadb_wasm PROPERTIES
    OUTPUT_NAME "pistadb"
    SUFFIX      ".js"
)

# ── Linker / Emscripten flags ─────────────────────────────────────────────────
target_link_options(pistadb_wasm PRIVATE
    # Embind: C++ ↔ JS type bindings
    -lembind

    # Exception support (must match compile flag)
    -fwasm-exceptions

    # Wrap output in a factory function:
    #   import PistaDB from './pistadb.js';
    #   const M = await PistaDB();
    -s MODULARIZE=1
    -s EXPORT_NAME='PistaDB'

    # Allow the WASM heap to grow as the database expands
    -s ALLOW_MEMORY_GROWTH=1

    # Starting heap (increase for large datasets)
    -s INITIAL_MEMORY=33554432   # 32 MB

    # Emscripten virtual filesystem — required for file I/O
    -s FORCE_FILESYSTEM=1

    # Expose FS and IDBFS to JavaScript for advanced filesystem control
    # (e.g. mounting IndexedDB for persistent browser storage)
    "-s EXPORTED_RUNTIME_METHODS=[\"FS\",\"IDBFS\"]"

    # Optimise output size and speed
    -O3
)

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "pistadb_wasm build configured")
message(STATUS "  Output : ${CMAKE_BINARY_DIR}/pistadb.js + pistadb.wasm")
message(STATUS "  Tip    : set INITIAL_MEMORY for large database workloads")
