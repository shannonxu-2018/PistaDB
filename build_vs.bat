@echo off
echo === PistaDB Visual Studio Build ===

set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize VS environment
    exit /b 1
)

set "SRC=C:\projects\github\PistaDB\src"
set "OUTDIR=C:\projects\github\PistaDB\build\Release"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "CFLAGS=/nologo /W4 /O2 /wd4200 /wd4996 /D_CRT_SECURE_NO_WARNINGS /DPISTADB_HAS_AVX2 /I%SRC% /MD"
set "LDFLAGS=/DLL /nologo /OPT:REF /OPT:ICF"

echo Compiling scalar sources...
cl %CFLAGS% /c /Fo"%OUTDIR%\distance.obj"        "%SRC%\distance.c" || exit /b 1
cl %CFLAGS% /arch:AVX2 /c /Fo"%OUTDIR%\distance_avx2.obj" "%SRC%\distance_avx2.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\utils.obj"            "%SRC%\utils.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\storage.obj"          "%SRC%\storage.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_linear.obj"     "%SRC%\index_linear.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_hnsw.obj"       "%SRC%\index_hnsw.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_ivf.obj"        "%SRC%\index_ivf.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_ivf_pq.obj"     "%SRC%\index_ivf_pq.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_diskann.obj"    "%SRC%\index_diskann.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_lsh.obj"        "%SRC%\index_lsh.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_scann.obj"      "%SRC%\index_scann.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\index_sq.obj"         "%SRC%\index_sq.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\pistadb.obj"          "%SRC%\pistadb.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\pistadb_batch.obj"    "%SRC%\pistadb_batch.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\pistadb_cache.obj"    "%SRC%\pistadb_cache.c" || exit /b 1
cl %CFLAGS% /c /Fo"%OUTDIR%\pistadb_txn.obj"      "%SRC%\pistadb_txn.c" || exit /b 1

echo Linking pistadb.dll...
link %LDFLAGS% /DEF:"C:\projects\github\PistaDB\build\CMakeFiles\pistadb.dir\exports.def" /OUT:"%OUTDIR%\pistadb.dll" ^
    "%OUTDIR%\distance.obj" ^
    "%OUTDIR%\distance_avx2.obj" ^
    "%OUTDIR%\utils.obj" ^
    "%OUTDIR%\storage.obj" ^
    "%OUTDIR%\index_linear.obj" ^
    "%OUTDIR%\index_hnsw.obj" ^
    "%OUTDIR%\index_ivf.obj" ^
    "%OUTDIR%\index_ivf_pq.obj" ^
    "%OUTDIR%\index_diskann.obj" ^
    "%OUTDIR%\index_lsh.obj" ^
    "%OUTDIR%\index_scann.obj" ^
    "%OUTDIR%\index_sq.obj" ^
    "%OUTDIR%\pistadb.obj" ^
    "%OUTDIR%\pistadb_batch.obj" ^
    "%OUTDIR%\pistadb_cache.obj" ^
    "%OUTDIR%\pistadb_txn.obj" ^
    kernel32.lib user32.lib || exit /b 1

copy /Y "%OUTDIR%\pistadb.dll" "C:\projects\github\PistaDB\libs\windows\x64\" >nul 2>&1
mkdir "C:\projects\github\PistaDB\libs\windows\x64" 2>nul
copy /Y "%OUTDIR%\pistadb.dll" "C:\projects\github\PistaDB\libs\windows\x64\" >nul

echo.
echo Build SUCCESS. Output: %OUTDIR%\pistadb.dll
exit /b 0
