#    ___ _    _        ___  ___
#   | _ (_)__| |_ __ _|   \| _ )
#   |  _/ (_-<  _/ _` | |) | _ \
#   |_| |_/__/\__\__,_|___/|___/

cmake_minimum_required(VERSION 3.18.1)
project(pistadb_android VERSION 1.0.0 LANGUAGES C)

# ── Compiler settings ──────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O3 -ffast-math)

# ── Source paths ───────────────────────────────────────────────────────────
# CMakeLists.txt lives in android/; PistaDB C sources are in ../src/
set(PISTADB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

set(PISTADB_C_SOURCES
    ${PISTADB_SRC}/distance.c
    ${PISTADB_SRC}/utils.c
    ${PISTADB_SRC}/storage.c
    ${PISTADB_SRC}/index_linear.c
    ${PISTADB_SRC}/index_hnsw.c
    ${PISTADB_SRC}/index_ivf.c
    ${PISTADB_SRC}/index_ivf_pq.c
    ${PISTADB_SRC}/index_diskann.c
    ${PISTADB_SRC}/index_lsh.c
    ${PISTADB_SRC}/index_scann.c
    ${PISTADB_SRC}/pistadb.c
)

# ── Shared library: pistadb_jni ────────────────────────────────────────────
# Compiled as a single .so that contains both the core PistaDB logic and
# the JNI bridge, so the app only needs to load one library.
add_library(pistadb_jni SHARED
    ${PISTADB_C_SOURCES}
    src/main/cpp/pistadb_jni.c
)

target_include_directories(pistadb_jni PRIVATE
    ${PISTADB_SRC}
)

# Link Android system libraries
find_library(LOG_LIB log)
target_link_libraries(pistadb_jni
    ${LOG_LIB}
    m          # math (required on Android too)
)

target_compile_definitions(pistadb_jni PRIVATE
    _ANDROID
    _CRT_SECURE_NO_WARNINGS   # suppress sprintf/fopen deprecation noise on some NDK versions
)
