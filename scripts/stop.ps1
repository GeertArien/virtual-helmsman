# Stop the voice agent backend + dashboard started by scripts\start.ps1.
#
#   .\scripts\stop.ps1
#   .\scripts\stop.ps1 -BackendOnly
#
# Processes are found by WHO OWNS THE PORT, not only by the recorded PID:
# the Config page's "Reload backend" replaces the backend process (exec), so
# the PID on file goes stale while the port keeps serving. Each process is
# killed with its whole tree (the frontend is a cmd -> npm -> node chain).
#
# If PowerShell refuses to load this ("running scripts is disabled"), use the
# scripts\stop.cmd wrapper, which bypasses the execution policy for this one
# script. Works in Windows PowerShell 5.1 and PowerShell 7.
param(
    [int]$ApiPort = 8765,
    [int]$FrontendPort = 5173,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $root 'logs\run'

function Stop-Tree([int]$procId, [string]$what) {
    if (-not $procId) { return $false }
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { return $false }
    # Via cmd with both streams swallowed: in Windows PowerShell 5.1 a native
    # command's stderr becomes a terminating error under EAP=Stop (even
    # redirected), and taskkill prints one for the harmless race where a
    # child already died with its parent.
    cmd /c "taskkill /PID $procId /T /F >nul 2>&1"
    Write-Host "${what}: stopped PID $procId (and children)"
    return $true
}

function Stop-Component([string]$name, [int]$port) {
    $stopped = $false

    # 1. Whoever listens on the port -- survives PID drift from exec reloads.
    $owners = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue).OwningProcess |
        Sort-Object -Unique
    foreach ($owner in $owners) {
        $stopped = (Stop-Tree $owner "$name (port $port)") -or $stopped
    }

    # 2. The recorded PID, if it still exists and was not the port owner
    #    (e.g. the cmd/npm wrapper above the node process, or a process
    #    still starting up that has not bound the port yet).
    $pidFile = Join-Path $runDir "$name.pid"
    if (Test-Path $pidFile) {
        $recorded = [int](Get-Content $pidFile)
        if ($owners -notcontains $recorded) {
            $stopped = (Stop-Tree $recorded "$name (pid file)") -or $stopped
        }
        Remove-Item $pidFile -Force
    }

    if (-not $stopped) { Write-Host "${name}: nothing running" }
}

if (-not $FrontendOnly) { Stop-Component 'backend' $ApiPort }
if (-not $BackendOnly) { Stop-Component 'frontend' $FrontendPort }
