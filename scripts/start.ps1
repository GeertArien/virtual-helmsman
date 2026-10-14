# Start the voice agent backend + SvelteKit dashboard as background processes.
#
#   .\scripts\start.ps1                  # config.local.yaml if present, else config.yaml
#   .\scripts\start.ps1 -Config config.examples\config.whisper.yaml
#   .\scripts\start.ps1 -BackendOnly
#
# Each process logs to logs\run\ and records its PID there. A component whose
# port is already listening is left alone, so the script is safe to re-run.
# Stop everything with scripts\stop.ps1.
#
# If PowerShell refuses to load this ("running scripts is disabled"), use the
# scripts\start.cmd wrapper, which bypasses the execution policy for this one
# script. Works in Windows PowerShell 5.1 and PowerShell 7.
param(
    [string]$Config,
    [int]$ApiPort = 8765,
    [int]$FrontendPort = 5173,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $root 'logs\run'
New-Item -ItemType Directory -Force $runDir | Out-Null

if (-not $Config) {
    $Config = if (Test-Path (Join-Path $root 'config.local.yaml')) { 'config.local.yaml' } else { 'config.yaml' }
}

function Test-PortListening([int]$port) {
    $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-ForHttp([string]$url, [int]$timeoutS, [string]$what) {
    $deadline = (Get-Date).AddSeconds($timeoutS)
    while ((Get-Date) -lt $deadline) {
        try {
            $res = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($res.StatusCode -eq 200) { return $true }
        } catch { }
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 2
    }
    Write-Host ''
    Write-Warning "$what did not answer at $url within ${timeoutS}s -- check the logs in logs\run\"
    return $false
}

# --- backend (voice agent + control-plane API) ------------------------------
if (-not $FrontendOnly) {
    if (Test-PortListening $ApiPort) {
        Write-Host "backend: already listening on port $ApiPort -- leaving it alone"
    } else {
        $python = Join-Path $root '.venv\Scripts\python.exe'
        if (-not (Test-Path $python)) {
            throw "No venv at .venv\ -- create it first (see README 'Installation')."
        }
        Write-Host "backend: starting (config: $Config) -- model loading takes ~10-30s"
        $p = Start-Process -FilePath $python `
            -ArgumentList '-m', 'voice_agent.main', '--config', $Config `
            -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $runDir 'backend.out.log') `
            -RedirectStandardError (Join-Path $runDir 'backend.err.log') `
            -WindowStyle Hidden -PassThru
        $p.Id | Set-Content (Join-Path $runDir 'backend.pid')
        if (Wait-ForHttp "http://127.0.0.1:$ApiPort/api/health" 120 'backend') {
            Write-Host "`nbackend: up on http://127.0.0.1:$ApiPort (PID $($p.Id))"
        }
    }
}

# --- frontend (SvelteKit dev server) ----------------------------------------
if (-not $BackendOnly) {
    if (Test-PortListening $FrontendPort) {
        Write-Host "frontend: already listening on port $FrontendPort -- leaving it alone"
    } else {
        $frontendDir = Join-Path $root 'frontend'
        if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
            Write-Host 'frontend: node_modules missing -- running npm install first'
            Push-Location $frontendDir
            try { npm install } finally { Pop-Location }
        }
        Write-Host 'frontend: starting dev server'
        # Via cmd so the npm.cmd shim resolves; stop.ps1 kills the whole tree.
        $p = Start-Process -FilePath 'cmd.exe' `
            -ArgumentList '/c', 'npm', 'run', 'dev' `
            -WorkingDirectory $frontendDir `
            -RedirectStandardOutput (Join-Path $runDir 'frontend.out.log') `
            -RedirectStandardError (Join-Path $runDir 'frontend.err.log') `
            -WindowStyle Hidden -PassThru
        $p.Id | Set-Content (Join-Path $runDir 'frontend.pid')
        if (Wait-ForHttp "http://localhost:$FrontendPort/" 60 'frontend') {
            Write-Host "`nfrontend: up on http://localhost:$FrontendPort (PID $($p.Id))"
        }
    }
}

Write-Host "`nDashboard: http://localhost:$FrontendPort  (API: http://127.0.0.1:$ApiPort)"
Write-Host 'Logs: logs\run\   Stop with: .\scripts\stop.ps1'
