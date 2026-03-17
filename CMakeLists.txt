cmake_minimum_required(VERSION 3.15)
project(PistaDB VERSION 1.0.0 LANGUAGES C)

# ── Compiler settings ─────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2)
    # Disable C4200 (flexible array) and C4996 (deprecation) warnings
    add_compile_options(/wd4200 /wd4996)
else()
    add_compile_options(-Wall -Wextra -O3 -ffast-math)
    if(NOT WIN32)
        # Link math library on Linux
        link_libraries(m)
    endif()
endif()

# ── SIMD source files (compiled with extra ISA flags) ─────────────────────
#
# distance_avx2.c  — requires AVX2 + FMA (x86-64, Haswell+)
# distance_neon.c  — requires NEON (AArch64 always has it; ARMv7 needs flag)
#
# We detect the host architecture at configure time and only compile the
# relevant SIMD file. The scalar fallback in distance.c is always compiled.

# ── Threading ─────────────────────────────────────────────────────────────
# pistadb_batch.c uses pthreads on POSIX and Win32 threads on Windows.
# CMake's Threads module gives us the correct link flags on every platform.
find_package(Threads REQUIRED)

include(CheckCCompilerFlag)

set(PISTADB_SIMD_SOURCES "")
set(PISTADB_SIMD_DEFS "")   # compile-time feature flags passed to distance.c

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
    if(MSVC)
        # MSVC enables AVX2 via /arch:AVX2; also links intrin.h
        check_c_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            set_source_files_properties(src/distance_avx2.c PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2")
            list(APPEND PISTADB_SIMD_SOURCES src/distance_avx2.c)
            list(APPEND PISTADB_SIMD_DEFS    PISTADB_HAS_AVX2)
            message(STATUS "  SIMD        : AVX2+FMA (MSVC /arch:AVX2)")
        endif()
    else()
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma"  COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
            set_source_files_properties(src/distance_avx2.c PROPERTIES
                COMPILE_OPTIONS "-mavx2;-mfma")
            list(APPEND PISTADB_SIMD_SOURCES src/distance_avx2.c)
            list(APPEND PISTADB_SIMD_DEFS    PISTADB_HAS_AVX2)
            message(STATUS "  SIMD        : AVX2+FMA (-mavx2 -mfma)")
        endif()
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # AArch64 always has NEON — no extra flag needed
    list(APPEND PISTADB_SIMD_SOURCES src/distance_neon.c)
    list(APPEND PISTADB_SIMD_DEFS    PISTADB_HAS_NEON)
    message(STATUS "  SIMD        : NEON (AArch64 built-in)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
    check_c_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        set_source_files_properties(src/distance_neon.c PROPERTIES
            COMPILE_OPTIONS "-mfpu=neon;-mfloat-abi=hard")
        list(APPEND PISTADB_SIMD_SOURCES src/distance_neon.c)
        list(APPEND PISTADB_SIMD_DEFS    PISTADB_HAS_NEON)
        message(STATUS "  SIMD        : NEON (-mfpu=neon)")
    endif()
else()
    message(STATUS "  SIMD        : scalar only (unrecognised arch)")
endif()

# ── Source files ──────────────────────────────────────────────────────────
set(PISTADB_SOURCES
    src/distance.c
    ${PISTADB_SIMD_SOURCES}
    src/utils.c
    src/storage.c
    src/index_linear.c
    src/index_hnsw.c
    src/index_ivf.c
    src/index_ivf_pq.c
    src/index_diskann.c
    src/index_lsh.c
    src/index_scann.c
    src/pistadb.c
    src/pistadb_batch.c
    src/pistadb_cache.c
    src/pistadb_txn.c
)

set(PISTADB_HEADERS
    src/pistadb_types.h
    src/pistadb.h
    src/pistadb_batch.h
    src/pistadb_cache.h
    src/pistadb_txn.h
    src/distance.h
    src/distance_simd.h
    src/utils.h
    src/storage.h
    src/index_linear.h
    src/index_hnsw.h
    src/index_ivf.h
    src/index_ivf_pq.h
    src/index_diskann.h
    src/index_lsh.h
    src/index_scann.h
)

# ── Shared library (for Python ctypes) ───────────────────────────────────
add_library(pistadb SHARED ${PISTADB_SOURCES} ${PISTADB_HEADERS})
target_include_directories(pistadb PUBLIC src)
set_target_properties(pistadb PROPERTIES
    OUTPUT_NAME "pistadb"
    VERSION     ${PROJECT_VERSION}
    SOVERSION   1
    # Make DLL importable from Python
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
    target_compile_definitions(pistadb PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
# Propagate SIMD feature flags so distance.c knows which kernels are linked
if(PISTADB_SIMD_DEFS)
    target_compile_definitions(pistadb PRIVATE ${PISTADB_SIMD_DEFS})
endif()

target_link_libraries(pistadb PRIVATE Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(pistadb m)
endif()

# ── Static library (for embedding) ───────────────────────────────────────
add_library(pistadb_static STATIC ${PISTADB_SOURCES} ${PISTADB_HEADERS})
target_include_directories(pistadb_static PUBLIC src)
set_target_properties(pistadb_static PROPERTIES OUTPUT_NAME "pistadb_static")
if(WIN32)
    target_compile_definitions(pistadb_static PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(PISTADB_SIMD_DEFS)
    target_compile_definitions(pistadb_static PRIVATE ${PISTADB_SIMD_DEFS})
endif()
target_link_libraries(pistadb_static PRIVATE Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(pistadb_static m)
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS pistadb pistadb_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
install(FILES ${PISTADB_HEADERS} DESTINATION include/pistadb)

# ── Print summary ─────────────────────────────────────────────────────────
message(STATUS "PistaDB ${PROJECT_VERSION}")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "  Output dir : ${CMAKE_BINARY_DIR}")
