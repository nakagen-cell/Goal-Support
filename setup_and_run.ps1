param(
    [string]$OpenAIKey,
    [int]$Port = 8000
)

Write-Host "== Goal Support System: Windows setup =="

# venv activate
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating venv..."
    python -m venv .venv
    . .venv\Scripts\Activate.ps1
}

python -m pip install --upgrade pip

# install deps
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} elseif (Test-Path "backend\requirements.txt") {
    pip install -r backend\requirements.txt
} else {
    Write-Error "No requirements file found. Expected requirements.txt or backend\requirements.txt"
    exit 1
}

# set key
$env:OPENAI_API_KEY = $OpenAIKey

# start uvicorn in background (PowerShell-safe)
Write-Host "Starting server on http://localhost:$Port ..."
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "backend.app:app --reload --port $Port"

# wait server startup
Start-Sleep -Seconds 3

# open browser
$url = "http://localhost:$Port/ui"
Start-Process $url

Write-Host ""
Write-Host "================================================="
Write-Host " Web UI:"
Write-Host "  $url"
Write-Host "================================================="
Write-Host ""
