$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"

function Find-CompatiblePython {
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @($pyLauncher.Source, "-3")
        }
    }

    $pythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        & $pythonCommand.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @($pythonCommand.Source)
        }
    }

    throw "Python 3.12 or newer was not found. Install it from https://www.python.org/downloads/windows/ and run this installer again."
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonCommand = @(Find-CompatiblePython)
    $pythonExecutable = $pythonCommand[0]
    $pythonArguments = @($pythonCommand | Select-Object -Skip 1)
    Write-Output "Creating OloHeart's isolated Python environment..."
    & $pythonExecutable @pythonArguments -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create OloHeart's .venv environment."
    }
}
else {
    & $pythonPath -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The existing .venv uses an unsupported Python version. Remove only the .venv folder and run setup.ps1 again."
    }
}

& $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "OloHeart's dependencies could not be installed. Check the internet connection and run setup.ps1 again."
}

& $pythonPath (Join-Path $projectRoot "run.py") --smoke-test
if ($LASTEXITCODE -ne 0) {
    throw "OloHeart was installed, but its startup validation failed."
}

Write-Output "OloHeart is ready. Run start-oloheart.cmd or double-click OloHeart.vbs."
