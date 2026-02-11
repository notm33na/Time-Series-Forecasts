# Start Forecast Pro Dash Server
# Run this script from the forecast-pro-dash directory

Write-Host "Starting Forecast Pro Dash Server..." -ForegroundColor Green
Write-Host ""

# Check if MongoDB is accessible
Write-Host "Checking MongoDB connection..." -ForegroundColor Yellow
try {
    python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000); client.server_info(); print('✓ MongoDB is accessible')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ MongoDB may not be running. Starting server anyway..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not verify MongoDB. Make sure it's running." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python -m backend.main

