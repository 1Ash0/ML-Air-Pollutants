param(
  [string]$PythonVersion = "3.11.9",
  [string]$RootDir = (Join-Path $PSScriptRoot "..\.py_std"),
  [string]$VenvDir = (Join-Path $PSScriptRoot "..\.venv_std")
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
  Write-Host ("[bootstrap] " + $msg)
}

Write-Step "Workspace: $(Get-Location)"
Write-Step "Target Python: $PythonVersion"

$pyDir = Join-Path $RootDir ("python-" + $PythonVersion + "-embed-amd64")
$zipPath = Join-Path $RootDir ("python-" + $PythonVersion + "-embed-amd64.zip")

New-Item -ItemType Directory -Force -Path $RootDir | Out-Null

if (-not (Test-Path $pyDir)) {
  $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
  Write-Step "Downloading embeddable Python from $url"
  # PowerShell Invoke-WebRequest frequently fails in constrained TLS setups.
  # Use the system Python's urllib instead (works reliably in this environment).
  python -c "import urllib.request,sys; url=sys.argv[1]; out=sys.argv[2]; print('downloading',url,'->',out); urllib.request.urlretrieve(url,out)" $url $zipPath
  Write-Step "Extracting to $pyDir"
  New-Item -ItemType Directory -Force -Path $pyDir | Out-Null
  Expand-Archive -Path $zipPath -DestinationPath $pyDir -Force
}

$pyExe = Join-Path $pyDir "python.exe"
if (-not (Test-Path $pyExe)) {
  throw "python.exe not found at $pyExe"
}

# Enable site-packages in embeddable Python by editing the ._pth file.
$pth = Get-ChildItem -Path $pyDir -Filter "python*._pth" | Select-Object -First 1
if ($pth) {
  $pthPath = $pth.FullName
  $lines = Get-Content -Path $pthPath
  $changed = $false
  $newLines = @()
  $foundImportSite = $false
  
  foreach ($line in $lines) {
    if ($line -match '^\s*#\s*import\s+site\s*$') {
      $newLines += "import site"
      $changed = $true
      $foundImportSite = $true
    } elseif ($line -match '^\s*import\s+site\s*$') {
      $newLines += "import site"
      $foundImportSite = $true
    } else {
      $newLines += $line
    }
  }
  
  if (-not $foundImportSite) {
    $newLines += "import site"
    $changed = $true
  }
  
  if ($changed) {
    Write-Step "Patching $pthPath to enable import site"
    Set-Content -Path $pthPath -Value $newLines -Encoding ASCII
  }
}

# Install pip into the embedded Python.
$getPip = Join-Path $RootDir "get-pip.py"
if (-not (Test-Path $getPip)) {
  Write-Step "Downloading get-pip.py"
  python -c "import urllib.request,sys; url=sys.argv[1]; out=sys.argv[2]; print('downloading',url,'->',out); urllib.request.urlretrieve(url,out)" "https://bootstrap.pypa.io/get-pip.py" $getPip
}

Write-Step "Installing pip into embedded Python"
& $pyExe $getPip | Out-Host

# Create a standard venv using the embedded Python.
if (Test-Path $VenvDir) {
  Write-Step "Removing existing venv at $VenvDir"
  Remove-Item -Recurse -Force $VenvDir
}

Write-Step "Creating venv at $VenvDir"
& $pyExe -m venv $VenvDir | Out-Host
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $VenvDir "Scripts\\python.exe"))) {
  Write-Step "Embedded Python lacks venv; using virtualenv instead"
  if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
  & $pyExe -m pip install --upgrade virtualenv | Out-Host
  & $pyExe -m virtualenv $VenvDir | Out-Host
}

$venvPy = Join-Path $VenvDir "Scripts\\python.exe"
if (-not (Test-Path $venvPy)) {
  throw "Venv python not found at $venvPy"
}

Write-Step "Upgrading pip in venv"
& $venvPy -m pip install --upgrade pip | Out-Host

Write-Step "Installing project requirements from requirements_ingest.txt"
& $venvPy -m pip install -r (Join-Path $PSScriptRoot "..\requirements_ingest.txt") | Out-Host

Write-Step "Done. Venv python: $venvPy"
