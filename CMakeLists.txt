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

# ── Source files ──────────────────────────────────────────────────────────
set(PISTADB_SOURCES
    src/distance.c
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
)

set(PISTADB_HEADERS
    src/pistadb_types.h
    src/pistadb.h
    src/distance.h
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

if(UNIX AND NOT APPLE)
    target_link_libraries(pistadb m)
endif()

# ── Static library (for embedding) ───────────────────────────────────────
add_library(pistadb_static STATIC ${PISTADB_SOURCES} ${PISTADB_HEADERS})
target_include_directories(pistadb_static PUBLIC src)
set_target_properties(pistadb_static PROPERTIES OUTPUT_NAME "pistadb_static")
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
