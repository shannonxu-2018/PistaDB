@echo off
REM PistaDB build script for Windows
REM Usage: build.bat [Release|Debug]

SET BUILD_TYPE=%1
IF "%BUILD_TYPE%"=="" SET BUILD_TYPE=Release

echo === PistaDB Build ===
echo Build type: %BUILD_TYPE%

cmake -B build -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
IF ERRORLEVEL 1 goto :error

cmake --build build --config %BUILD_TYPE%
IF ERRORLEVEL 1 goto :error

echo.
echo === Build complete ===
echo Shared library should be in build\%BUILD_TYPE%\pistadb.dll
echo.
echo To install the Python package:
echo   pip install -e wrap\python\
echo.
echo To run tests (PowerShell):
echo   $env:PISTADB_LIB_DIR="build\%BUILD_TYPE%"; pytest tests\ -v
echo.
echo To run the example:
echo   set PISTADB_LIB_DIR=build\%BUILD_TYPE%
echo   python examples\example.py
goto :end

:error
echo Build FAILED.
exit /b 1

:end
