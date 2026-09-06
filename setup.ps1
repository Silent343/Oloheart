$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv $environmentPath
}

& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Output "OloHeart is ready. Double-click OloHeart.vbs or run: .\.venv\Scripts\python.exe run.py"
