# PowerShell script to clear all MongoDB data
# Usage: .\clear_mongodb.ps1 [--force]

$scriptPath = Join-Path $PSScriptRoot "clear_mongodb.py"
$projectRoot = Split-Path (Split-Path $PSScriptRoot)

Push-Location $projectRoot

if ($args -contains "--force") {
    python $scriptPath --force
} else {
    python $scriptPath
}

Pop-Location

