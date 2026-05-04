@echo off
REM Backward-compatible forwarder. The real script lives at
REM   scripts\windows\build.bat
REM and additionally copies artifacts into libs\windows\x64\.
REM See scripts\README.md and INTEGRATION.md for the full per-OS layout.
@call "%~dp0scripts\windows\build.bat" %*
