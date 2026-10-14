@echo off
rem Execution-policy-proof wrapper for start.ps1 (also double-clickable).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
