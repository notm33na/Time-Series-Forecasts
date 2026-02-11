# Quick script to train a model in ForecastPro
# Usage: .\train_model.ps1 -Symbol "AAPL" -ModelType "LSTM" -Horizon "24h"

param(
    [string]$Symbol = "AAPL",
    [string]$ModelType = "Ensemble",
    [string]$Horizon = "24h",
    [string]$ApiUrl = "http://localhost:8000"
)

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "ForecastPro - Model Training Script" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if server is running
Write-Host "Step 1: Checking server connection..." -ForegroundColor Green
try {
    $health = Invoke-WebRequest -Uri "$ApiUrl/health" -UseBasicParsing -ErrorAction Stop
    Write-Host "✓ Server is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Server is not running! Please start the backend server first:" -ForegroundColor Red
    Write-Host "  cd backend" -ForegroundColor Yellow
    Write-Host "  python -m backend.main" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check if data exists, ingest if needed
Write-Host "`nStep 2: Checking for historical data..." -ForegroundColor Green
try {
    $prices = Invoke-WebRequest -Uri "$ApiUrl/api/data/prices/$Symbol?limit=10" -UseBasicParsing
    $priceData = $prices.Content | ConvertFrom-Json
    if ($priceData.prices.Count -eq 0) {
        Write-Host "No data found. Ingesting data..." -ForegroundColor Yellow
        $ingest = @{
            symbol = $Symbol
            period = "2y"
        } | ConvertTo-Json
        
        $ingestResult = Invoke-WebRequest -Uri "$ApiUrl/api/data/ingest" `
            -Method POST -ContentType "application/json" -Body $ingest -UseBasicParsing
        $ingestData = $ingestResult.Content | ConvertFrom-Json
        Write-Host "✓ Ingested $($ingestData.records_ingested) records" -ForegroundColor Green
    } else {
        Write-Host "✓ Data already exists ($($priceData.prices.Count) records found)" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Error checking/ingesting data: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Train the model
Write-Host "`nStep 3: Training $ModelType model..." -ForegroundColor Green
try {
    $train = @{
        symbol = $Symbol
        model_type = $ModelType
        horizon = $Horizon
    } | ConvertTo-Json
    
    Write-Host "Training in progress (this may take a minute)..." -ForegroundColor Yellow
    $trainResult = Invoke-WebRequest -Uri "$ApiUrl/api/forecast/train" `
        -Method POST -ContentType "application/json" -Body $train -UseBasicParsing
    $trainData = $trainResult.Content | ConvertFrom-Json
    
    Write-Host "✓ Model trained successfully!" -ForegroundColor Green
    Write-Host "  Model Version: $($trainData.version_tag)" -ForegroundColor Cyan
    Write-Host "  Model ID: $($trainData.model_version_id)" -ForegroundColor Cyan
    Write-Host "  Trained At: $($trainData.trained_at)" -ForegroundColor Cyan
} catch {
    Write-Host "✗ Error training model: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "  Response: $responseBody" -ForegroundColor Yellow
    }
    exit 1
}

# Step 4: Generate a forecast to verify
Write-Host "`nStep 4: Generating forecast to verify model..." -ForegroundColor Green
try {
    $forecast = @{
        symbol = $Symbol
        model_type = $ModelType
        horizon = $Horizon
    } | ConvertTo-Json
    
    $forecastResult = Invoke-WebRequest -Uri "$ApiUrl/api/forecast/run" `
        -Method POST -ContentType "application/json" -Body $forecast -UseBasicParsing
    $forecastData = $forecastResult.Content | ConvertFrom-Json
    
    Write-Host "✓ Forecast generated successfully!" -ForegroundColor Green
    
    if ($forecastData.predictions) {
        $predCount = $forecastData.predictions.Count
        Write-Host "  Predictions: $predCount values" -ForegroundColor Cyan
        Write-Host "  First 5 predictions: $($forecastData.predictions[0..4] -join ', ')" -ForegroundColor Yellow
    } elseif ($forecastData.ensemble_forecast) {
        $predCount = $forecastData.ensemble_forecast.Count
        Write-Host "  Ensemble predictions: $predCount values" -ForegroundColor Cyan
        Write-Host "  First 5 predictions: $($forecastData.ensemble_forecast[0..4] -join ', ')" -ForegroundColor Yellow
        if ($forecastData.metrics) {
            Write-Host "  Model Metrics:" -ForegroundColor Cyan
            foreach ($model in $forecastData.metrics.PSObject.Properties) {
                $metrics = $model.Value
                Write-Host "    $($model.Name): MAE=$($metrics.MAE), RMSE=$($metrics.RMSE), MAPE=$($metrics.MAPE)%" -ForegroundColor Yellow
            }
        }
    }
} catch {
    Write-Host "✗ Error generating forecast: $_" -ForegroundColor Red
    Write-Host "  (Model was trained successfully, but forecast generation failed)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "✅ Training Complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. View API documentation: $ApiUrl/docs" -ForegroundColor White
Write-Host "2. Check model metrics: $ApiUrl/api/evaluation/metrics/$Symbol" -ForegroundColor White
Write-Host "3. List all models: $ApiUrl/api/forecast/models/$Symbol" -ForegroundColor White
Write-Host "4. Use the frontend dashboard to visualize forecasts" -ForegroundColor White
Write-Host ""

