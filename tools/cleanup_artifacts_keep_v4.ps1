$ErrorActionPreference = "Stop"

$Root = "E:\ML Based Prediction Air Pollutants"
$Artifacts = Join-Path $Root "artifacts"
$Archive = Join-Path $Artifacts ("_archive_nonv4_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "Cleaning artifacts (keeping only final v4 outputs/models)..." -ForegroundColor Cyan
Write-Host "  Artifacts: $Artifacts"
Write-Host "  Archive  : $Archive"

New-Item -ItemType Directory -Path $Archive | Out-Null

# Keep-list patterns (names inside artifacts/)
$keepExact = @(
  "classical_metrics_routed_v4.json",
  "classical_metrics_per_station.json",
  "metrics.csv",
  "lstm_metrics.json",
  "feature_schema_v4.json",
  "standard_scaler_v4.json",
  "feature_splits_report_v4.json",
  "model_lstm.keras",
  "lstm_scalers.pkl"
)

$keepRegex = @(
  '^plots$',
  '^model_ridge_[A-Z0-9_]+\.pkl$',
  '^model_xgb_[A-Z0-9_]+\.json$',
  '^model_xgb_[A-Z0-9_]+_meta\.json$'
)

function Should-Keep([string]$name) {
  if ($keepExact -contains $name) { return $true }
  foreach ($rx in $keepRegex) {
    if ($name -match $rx) { return $true }
  }
  return $false
}

$items = Get-ChildItem $Artifacts -Force
foreach ($it in $items) {
  $name = $it.Name
  if ($name -like "_archive_nonv4_*") { continue }

  if (Should-Keep $name) {
    Write-Host "  KEEP  $name" -ForegroundColor Green
    continue
  }

  $dest = Join-Path $Archive $name
  Write-Host "  MOVE  $name -> $dest" -ForegroundColor Yellow
  Move-Item -LiteralPath $it.FullName -Destination $dest -Force
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Archived non-v4 artifacts at:" -ForegroundColor Cyan
Write-Host "  $Archive" -ForegroundColor Green

