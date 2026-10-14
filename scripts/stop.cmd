@echo off
rem Execution-policy-proof wrapper for stop.ps1 (also double-clickable).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
