$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    throw "GITHUB_ENV is not set; this helper is intended for GitHub Actions"
}

if ([string]::IsNullOrWhiteSpace($env:GITHUB_PATH)) {
    throw "GITHUB_PATH is not set; this helper is intended for GitHub Actions"
}

if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    throw "RUNNER_TEMP is not set; this helper is intended for GitHub Actions"
}

$vcpkg = $null
if (-not [string]::IsNullOrWhiteSpace($env:VCPKG_ROOT)) {
    $candidate = Join-Path $env:VCPKG_ROOT "vcpkg.exe"
    if (Test-Path -LiteralPath $candidate) {
        $vcpkg = $candidate
    }
}

if (-not $vcpkg) {
    $found = Get-Command vcpkg.exe -ErrorAction SilentlyContinue
    if ($found) {
        $vcpkg = $found.Source
    }
}

if (-not $vcpkg) {
    throw "vcpkg.exe was not found. Set VCPKG_ROOT or add vcpkg.exe to PATH."
}

$root = Join-Path $env:RUNNER_TEMP "cuda-oxide-vcpkg-libffi"
$manifestRoot = Join-Path $root "manifest"
$installRoot = Join-Path $root "installed"
New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
New-Item -ItemType Directory -Force -Path $installRoot | Out-Null

$manifest = [ordered]@{
    name = "cuda-oxide-ci-libffi"
    "version-string" = "0.0.0"
    dependencies = @("libffi")
} | ConvertTo-Json -Depth 4
$manifest | Set-Content -LiteralPath (Join-Path $manifestRoot "vcpkg.json") -Encoding utf8

& $vcpkg x-update-baseline --add-initial-baseline "--x-manifest-root=$manifestRoot"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $vcpkg install --triplet x64-windows "--x-manifest-root=$manifestRoot" "--x-install-root=$installRoot"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$packageRoot = Join-Path $installRoot "x64-windows"
$libDir = Join-Path $packageRoot "lib"
$binDir = Join-Path $packageRoot "bin"
$ffiLib = Join-Path $libDir "ffi.lib"
if (-not (Test-Path -LiteralPath $ffiLib)) {
    throw "vcpkg completed, but ffi.lib was not found at $ffiLib"
}

"LIBFFI_LIB_DIR=$libDir" | Out-File -FilePath $env:GITHUB_ENV -Append
"LIBFFI_BIN_DIR=$binDir" | Out-File -FilePath $env:GITHUB_ENV -Append
"LIB=$libDir;$env:LIB" | Out-File -FilePath $env:GITHUB_ENV -Append
$binDir | Out-File -FilePath $env:GITHUB_PATH -Append
