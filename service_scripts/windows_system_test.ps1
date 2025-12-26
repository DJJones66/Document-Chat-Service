param(
    [string]$CondaEnv = "BrainDriveDev",
    [string]$ApiHost = "127.0.0.1",
    [int]$PortStart = 18081,
    [int]$PortCount = 10
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$dataDir = Join-Path $repoRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

if ($CondaEnv -and (Get-Command conda -ErrorAction SilentlyContinue)) {
    try {
        $condaPython = (& conda run -n $CondaEnv python -c "import sys; print(sys.executable)" 2>$null).Trim()
        if ($condaPython) {
            $env:PYTHON_BIN = $condaPython
        }
    } catch {
        Write-Host "Warning: Unable to resolve conda python for $CondaEnv; falling back to PATH."
    }
}

$env:VENV_PATH = ".venv"

function Get-FreePort {
    param(
        [int]$Start,
        [int]$Count
    )
    for ($i = 0; $i -lt $Count; $i++) {
        $port = $Start + $i
        if (-not (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue)) {
            return $port
        }
    }
    throw "No free port found in range $Start-$($Start + $Count - 1)."
}

function Invoke-Python {
    param([string[]]$Args)
    if ($CondaEnv -and (Get-Command conda -ErrorAction SilentlyContinue)) {
        & conda run -n $CondaEnv python @Args
    } else {
        & python @Args
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Args -join ' ')"
    }
}

function Wait-ForHealth {
    param(
        [string]$Url,
        [int]$Retries = 12,
        [int]$DelaySeconds = 3
    )
    for ($i = 0; $i -lt $Retries; $i++) {
        try {
            return Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5
        } catch {
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    throw "Health check failed at $Url"
}

$port = Get-FreePort -Start $PortStart -Count $PortCount
$env:API_HOST = $ApiHost
$env:API_PORT = $port
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_SYMLINKS = "1"

Push-Location $repoRoot
try {
    Invoke-Python @(".\\service_scripts\\create_venv.py")
    Invoke-Python @(".\\service_scripts\\install_with_venv.py")

    $venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Venv python not found at $venvPython."
    }

    $startStdout = Join-Path $dataDir "test_start_stdout.log"
    $startStderr = Join-Path $dataDir "test_start_stderr.log"
    $startProc = Start-Process -FilePath $venvPython -ArgumentList @(".\\service_scripts\\start_with_venv.py") -WorkingDirectory $repoRoot -PassThru -NoNewWindow -RedirectStandardOutput $startStdout -RedirectStandardError $startStderr

    $healthUrl = "http://$ApiHost`:$port/health"
    Wait-ForHealth -Url $healthUrl | Out-Null

    $restartStdout = Join-Path $dataDir "test_restart_stdout.log"
    $restartStderr = Join-Path $dataDir "test_restart_stderr.log"
    $restartProc = Start-Process -FilePath $venvPython -ArgumentList @(".\\service_scripts\\restart_with_venv.py") -WorkingDirectory $repoRoot -PassThru -NoNewWindow -RedirectStandardOutput $restartStdout -RedirectStandardError $restartStderr

    Wait-ForHealth -Url $healthUrl | Out-Null

    & $venvPython ".\\service_scripts\\shutdown_with_venv.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Shutdown failed with exit code $LASTEXITCODE."
    }

    Start-Sleep -Seconds 2
    if (Get-Process -Id $startProc.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $startProc.Id -ErrorAction SilentlyContinue
    }
    if (Get-Process -Id $restartProc.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $restartProc.Id -ErrorAction SilentlyContinue
    }

    Write-Host "System test OK on $healthUrl"
    Write-Host "Logs: $startStderr, $restartStderr"
} finally {
    Pop-Location
}
