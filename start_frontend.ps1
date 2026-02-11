# Start Forecast Pro Dash Frontend
# Run this script from the forecast-pro-dash directory

Write-Host "Starting Forecast Pro Dash Frontend..." -ForegroundColor Green
Write-Host ""

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host ""
}

Write-Host "Starting Vite development server..." -ForegroundColor Cyan
Write-Host "Frontend will be available at http://localhost:5173 (or check output below)" -ForegroundColor Yellow
Write-Host "Backend API is at: http://localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press CTRL+C to stop the frontend server" -ForegroundColor Yellow
Write-Host ""

# Start the frontend
npm run dev

