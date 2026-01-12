# Streamlit Startup Script for POLARIS Hypothesis Agent (Optimized)
# Set API keys as environment variables
$env:GOOGLE_API_KEY = "AIzaSyAcd3kZ7NM7JZ27R13IMU2rHU2tSbUTShw"
$env:GEMINI_API_KEY = "AIzaSyAcd3kZ7NM7JZ27R13IMU2rHU2tSbUTShw"

# Change to script directory (polaris_ahmadi)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "========================================"
Write-Host "POLARIS Hypothesis Agent - Streamlit"
Write-Host "========================================"
Write-Host "Current directory: $(Get-Location)"
Write-Host "API Key: $($env:GOOGLE_API_KEY.Substring(0, [Math]::Min(10, $env:GOOGLE_API_KEY.Length)))..."
Write-Host ""

# Verify file exists
$streamlitFile = Join-Path $scriptDir "streamlit_app.py"
if (Test-Path $streamlitFile) {
    Write-Host "Found streamlit_app.py at: $streamlitFile"
    Write-Host "Starting Streamlit app (optimized for speed)..."
    Write-Host ""

    # Performance optimizations for faster startup
    $env:STREAMLIT_SERVER_HEADLESS = "true"
    $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
    $env:STREAMLIT_CLIENT_SHOW_SIDEBAR_NAVIGATION = "false"

    # Run streamlit with optimizations
    python -m streamlit run $streamlitFile --server.headless true --browser.gatherUsageStats false
} else {
    Write-Host "Error: streamlit_app.py not found!"
    Write-Host "Expected location: $streamlitFile"
    Write-Host "Current directory: $(Get-Location)"
    Write-Host ""
    Write-Host "Please ensure you are running this script from the polaris_ahmadi directory."
}