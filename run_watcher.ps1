# PowerShell script to start the Watcher Server
# Run this in a separate terminal window

Write-Host "Starting POLARIS Watcher Server..." -ForegroundColor Green
Write-Host ""

# Change to the polaris_ahmadi directory
Set-Location -Path "$PSScriptRoot\polaris_ahmadi"

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "ERROR: Python not found. Please install Python or add it to PATH." -ForegroundColor Red
    exit 1
}

# Check if required packages are installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan
try {
    python -c "import fastapi, uvicorn, watchdog" 2>&1 | Out-Null
    Write-Host "✅ All dependencies are installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Missing dependencies. Installing..." -ForegroundColor Yellow
    pip install fastapi uvicorn watchdog
}

# Check for API key in environment
$apiKey = $env:HUGGINGFACE_API_KEY
if (-not $apiKey) {
    $apiKey = $env:HF_API_KEY
}
if (-not $apiKey) {
    $apiKey = $env:LLM_API_KEY
}

if (-not $apiKey) {
    Write-Host ""
    Write-Host "⚠️  WARNING: No API key found in environment variables!" -ForegroundColor Yellow
    Write-Host "   Set HUGGINGFACE_API_KEY (or HF_API_KEY) or LLM_API_KEY before starting." -ForegroundColor Yellow
    Write-Host "   Example: `$env:HUGGINGFACE_API_KEY = 'your-api-key-here'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   The watcher will start but LLM features may not work." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        exit 1
    }
} else {
    Write-Host "✓ API key found in environment" -ForegroundColor Green
}

# Start the watcher server
Write-Host ""
Write-Host "Starting watcher server on port 8000..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m watcher.server
