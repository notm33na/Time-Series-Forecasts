# PowerShell script to populate database with historical data
# Usage: .\populate_data.ps1 -Symbol AAPL -Period 5y -Interval 1d

param(
    [Parameter(Mandatory=$true)]
    [string]$Symbol,
    
    [Parameter(Mandatory=$false)]
    [string]$Period = "5y",
    
    [Parameter(Mandatory=$false)]
    [string]$Interval = "1d"
)

$apiUrl = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Populating Database with Historical Data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Symbol: $Symbol" -ForegroundColor Yellow
Write-Host "Period: $Period" -ForegroundColor Yellow
Write-Host "Interval: $Interval" -ForegroundColor Yellow
Write-Host ""

# Check if backend is running
try {
    $healthCheck = Invoke-WebRequest -Uri "$apiUrl/docs" -Method GET -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Backend server is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend server is not running. Please start it first:" -ForegroundColor Red
    Write-Host "  cd backend" -ForegroundColor Yellow
    Write-Host "  python -m backend.main" -ForegroundColor Yellow
    exit 1
}

# Prepare request body
$body = @{
    symbol = $Symbol.ToUpper()
    period = $Period
    interval = $Interval
} | ConvertTo-Json

Write-Host "Ingesting data..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "$apiUrl/api/data/ingest" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -UseBasicParsing
    
    $result = $response.Content | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "✓ Success!" -ForegroundColor Green
    Write-Host "  Records ingested: $($result.records_ingested)" -ForegroundColor Green
    Write-Host "  Symbol: $($result.symbol)" -ForegroundColor Green
    Write-Host "  Interval: $($result.interval)" -ForegroundColor Green
    Write-Host ""
    
    # Verify data
    Write-Host "Verifying data..." -ForegroundColor Cyan
    $verifyResponse = Invoke-WebRequest -Uri "$apiUrl/api/data/prices/$Symbol?limit=10" -UseBasicParsing
    $verifyData = $verifyResponse.Content | ConvertFrom-Json
    
    if ($verifyData.prices.Count -gt 0) {
        Write-Host "✓ Data verified: $($verifyData.prices.Count) records available (showing first 10)" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now increase the limit in Dashboard.tsx to show more data points!" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host ""
    Write-Host "✗ Error ingesting data:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
    
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan

