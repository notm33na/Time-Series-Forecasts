# PowerShell script to reset and retrain all AAPL models
# This script deletes all AAPL artifacts, data, and retrains all models

$Symbol = "AAPL"
$apiUrl = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RESET AND RETRAIN AAPL MODELS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  WARNING: This will permanently delete:" -ForegroundColor Yellow
Write-Host "   • All AAPL model files from filesystem (artifacts directory)" -ForegroundColor Yellow
Write-Host "   • All AAPL data from MongoDB (prices, models, forecasts, metrics)" -ForegroundColor Yellow
Write-Host "   • Then retrain all models for AAPL" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Are you sure you want to reset and retrain AAPL? (yes/no)"
if ($confirm -ne "yes" -and $confirm -ne "y") {
    Write-Host "❌ Operation cancelled." -ForegroundColor Red
    exit 0
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 1: DELETING AAPL ARTIFACTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if artifacts directory exists
$artifactsDir = "backend\artifacts"
if (-not (Test-Path $artifactsDir)) {
    Write-Host "⚠️  Artifacts directory does not exist: $artifactsDir" -ForegroundColor Yellow
} else {
    # Find and delete AAPL files
    $aaplFiles = Get-ChildItem -Path $artifactsDir -Filter "*AAPL*" -File
    if ($aaplFiles.Count -eq 0) {
        Write-Host "✓ No AAPL model files found" -ForegroundColor Green
    } else {
        Write-Host "Found $($aaplFiles.Count) AAPL files" -ForegroundColor Cyan
        foreach ($file in $aaplFiles) {
            Remove-Item $file.FullName -Force
            Write-Host "   ✓ Deleted: $($file.Name)" -ForegroundColor Green
        }
        Write-Host "✅ Deleted $($aaplFiles.Count) AAPL files from filesystem" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 2: DELETING AAPL DATA FROM MONGODB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if backend is running
try {
    $healthCheck = Invoke-WebRequest -Uri "$apiUrl/docs" -Method GET -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Backend server is running" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend server is not running. MongoDB deletion will be skipped." -ForegroundColor Yellow
    Write-Host "   You can manually delete AAPL data from MongoDB later." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Note: To delete from MongoDB directly, you can use:" -ForegroundColor Cyan
    Write-Host "   python scripts/reset_and_retrain_aapl.py" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 3: RETRAINING ALL MODELS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Run the Python script to retrain
Write-Host "Running Python script to retrain all models..." -ForegroundColor Cyan
Write-Host ""

$pythonScript = "scripts\reset_and_retrain_aapl.py"
if (-not (Test-Path $pythonScript)) {
    Write-Host "❌ Python script not found: $pythonScript" -ForegroundColor Red
    Write-Host "   Please run the Python script directly:" -ForegroundColor Yellow
    Write-Host "   python scripts/reset_and_retrain_aapl.py" -ForegroundColor Yellow
    exit 1
}

try {
    python $pythonScript
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ All operations completed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ Some operations failed. Check the output above." -ForegroundColor Red
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error running Python script: $_" -ForegroundColor Red
    Write-Host "   Please run manually: python scripts/reset_and_retrain_aapl.py" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

