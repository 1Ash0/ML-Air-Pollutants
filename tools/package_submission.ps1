$ErrorActionPreference = "Stop"

$Root = "E:\ML Based Prediction Air Pollutants"
$Artifacts = Join-Path $Root "artifacts"
$Out = Join-Path $Artifacts ("SUBMISSION_BUNDLE_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "Packaging submission bundle..." -ForegroundColor Cyan
Write-Host "  Root: $Root"
Write-Host "  Out : $Out"

New-Item -ItemType Directory -Path $Out | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Out "plots") | Out-Null

function Copy-IfExists([string]$Path, [string]$DestDir) {
    if (Test-Path $Path) {
        Copy-Item $Path $DestDir -Force
        Write-Host "  + $Path" -ForegroundColor Green
    } else {
        Write-Host "  ! Missing: $Path" -ForegroundColor Yellow
    }
}

# Core report artifacts (small)
Copy-IfExists (Join-Path $Artifacts "metrics.csv") $Out
Copy-IfExists (Join-Path $Artifacts "classical_metrics_routed_v4.json") $Out
Copy-IfExists (Join-Path $Artifacts "classical_metrics_per_station.json") $Out
Copy-IfExists (Join-Path $Artifacts "lstm_metrics.json") $Out

# Plots
$PlotsDir = Join-Path $Artifacts "plots"
if (Test-Path $PlotsDir) {
    Get-ChildItem $PlotsDir -Filter *.png | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $Out "plots") -Force
        Write-Host "  + plots/$($_.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "  ! Missing plots dir: $PlotsDir" -ForegroundColor Yellow
}

# Code files needed for submission/repro
$CodeFiles = @(
    ".gitignore",
    "README.md",
    "PHASES_RUNBOOK.md",
    "IMPLEMENTATION_STATUS.md",
    "PROJECT_BLUEPRINT.md",
    "1_ingest_excel.py",
    "2_preprocess_and_features.py",
    "3_train_classical.py",
    "4_train_lstm.py",
    "5_evaluate_and_plot.py",
    "9_route_models_by_station.py",
    "requirements_ingest.txt",
    "requirements_ml.txt"
)

foreach ($f in $CodeFiles) {
    Copy-IfExists (Join-Path $Root $f) $Out
}

Write-Host ""
Write-Host "Done. Bundle created at:" -ForegroundColor Cyan
Write-Host "  $Out" -ForegroundColor Green

