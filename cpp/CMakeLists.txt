# cpp/CMakeLists.txt — CMake integration for the pistadb C++ header-only binding.
#
# Usage (from your project's CMakeLists.txt):
#
#   add_subdirectory(path/to/PistaDB)          # builds pistadb shared lib
#   add_subdirectory(path/to/PistaDB/cpp)      # registers pistadb_cpp interface target
#
#   target_link_libraries(my_app PRIVATE pistadb_cpp)
#
# That's it. pistadb.hpp and the C headers are added to your include path automatically.

cmake_minimum_required(VERSION 3.14)
project(pistadb_cpp LANGUAGES CXX)

# ── Interface library (header-only, nothing to compile) ───────────────────────
add_library(pistadb_cpp INTERFACE)

target_compile_features(pistadb_cpp INTERFACE cxx_std_17)

target_include_directories(pistadb_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>        # exposes pistadb.hpp
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src> # exposes pistadb.h (C API)
    $<INSTALL_INTERFACE:include>
)

# Link the native C library.
# The 'pistadb' target is defined in the root CMakeLists.txt.
# If using this directory standalone, set PISTADB_LIB_DIR and PISTADB_INCLUDE_DIR.
if(TARGET pistadb)
    target_link_libraries(pistadb_cpp INTERFACE pistadb)
else()
    find_library(PISTADB_LIB pistadb
        HINTS ${PISTADB_LIB_DIR} ENV PISTADB_LIB_DIR)
    if(NOT PISTADB_LIB)
        message(WARNING
            "pistadb library not found. Set PISTADB_LIB_DIR to the directory "
            "containing pistadb.dll / libpistadb.so / libpistadb.dylib.")
    else()
        target_link_libraries(pistadb_cpp INTERFACE ${PISTADB_LIB})
    endif()
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "pistadb_cpp: C++17 header-only wrapper configured")
message(STATUS "  Header: ${CMAKE_CURRENT_SOURCE_DIR}/pistadb.hpp")
