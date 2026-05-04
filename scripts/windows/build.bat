@echo off
REM ============================================================================
REM  PistaDB - Windows build script (MSVC)
REM
REM  Usage:
REM    scripts\windows\build.bat            (defaults to Release)
REM    scripts\windows\build.bat Debug
REM
REM  Output:
REM    libs\windows\x64\pistadb.dll
REM    libs\windows\x64\pistadb.lib          (DLL import library)
REM    libs\windows\x64\pistadb_static.lib   (static library)
REM
REM  Requirements:
REM    - Visual Studio 2019+ with C++ build tools (MSVC), or VS Build Tools
REM    - CMake 3.15+
REM    - Run from a "Developer Command Prompt for VS" if cl.exe is not on PATH
REM ============================================================================

setlocal EnableDelayedExpansion

set "BUILD_TYPE=%~1"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"

REM Resolve repo root: this script lives at <root>\scripts\windows\build.bat
set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%" >nul
set "REPO_ROOT=%CD%"
popd >nul

set "BUILD_DIR=%REPO_ROOT%\build"
set "OUT_DIR=%REPO_ROOT%\libs\windows\x64"

echo === PistaDB Windows Build ===
echo Repo root  : %REPO_ROOT%
echo Build type : %BUILD_TYPE%
echo Build dir  : %BUILD_DIR%
echo Output dir : %OUT_DIR%
echo.

cmake -B "%BUILD_DIR%" -S "%REPO_ROOT%" ^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
if errorlevel 1 goto :err

cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 goto :err

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM MSVC's Visual Studio generator places artifacts under build\<Config>\,
REM but the Makefile / Ninja generators place them directly under build\.
REM Resolve whichever layout the active CMake cache produced.
set "SRC_DIR=%BUILD_DIR%\%BUILD_TYPE%"
if not exist "%SRC_DIR%\pistadb.dll" set "SRC_DIR=%BUILD_DIR%"

if not exist "%SRC_DIR%\pistadb.dll" (
    echo ERROR: pistadb.dll not found under "%BUILD_DIR%\%BUILD_TYPE%\" or "%BUILD_DIR%\".
    goto :err
)

copy /Y "%SRC_DIR%\pistadb.dll"        "%OUT_DIR%\" >nul
if errorlevel 1 goto :err
if exist "%SRC_DIR%\pistadb.lib"        copy /Y "%SRC_DIR%\pistadb.lib"        "%OUT_DIR%\" >nul
if exist "%SRC_DIR%\pistadb_static.lib" copy /Y "%SRC_DIR%\pistadb_static.lib" "%OUT_DIR%\" >nul

echo.
echo === Build complete ===
echo Artifacts:
dir /B "%OUT_DIR%"
echo.
echo To use from Python:
echo   set PISTADB_LIB_DIR=%OUT_DIR%
echo   python -c "import pistadb; print(pistadb.__file__)"
echo.
echo To run tests:
echo   set PISTADB_LIB_DIR=%OUT_DIR%
echo   set PYTHONPATH=%REPO_ROOT%\wrap\python
echo   pytest tests\ -v
exit /b 0

:err
echo.
echo Build FAILED.
exit /b 1
