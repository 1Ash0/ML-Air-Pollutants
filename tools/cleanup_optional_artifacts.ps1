$ErrorActionPreference = "Stop"

$Root = "E:\ML Based Prediction Air Pollutants"
$Artifacts = Join-Path $Root "artifacts"

Write-Host "Cleaning optional artifacts (safe to regenerate)..." -ForegroundColor Cyan

$patterns = @(
    "viva_metrics_*.csv",
    "viva_metrics_*.json",
    "metrics_legacy*.csv",
    "metrics_by_station_legacy*.csv",
    "plots_legacy"
)

foreach ($p in $patterns) {
    $path = Join-Path $Artifacts $p
    Get-ChildItem $path -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Removing $($_.FullName)" -ForegroundColor Yellow
        Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Done." -ForegroundColor Green

