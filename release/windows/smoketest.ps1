param(
    [switch]$RequireCargo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cargoOxide = Join-Path $root "cargo-oxide.exe"
$backend = Join-Path $root "rustc_codegen_cuda.dll"

function Write-Pass {
    param([string]$Name)
    Write-Host "SMOKE PASS: $Name"
}

function Write-Fail {
    param([string]$Name, [string]$Reason)
    Write-Host "SMOKE FAIL: $Name - $Reason"
    exit 1
}

function Write-Skip {
    param([string]$Name, [string]$Reason)
    Write-Host "SMOKE SKIP: $Name - $Reason"
}

if (-not (Test-Path -LiteralPath $cargoOxide)) {
    Write-Fail "archive layout" "cargo-oxide.exe is missing"
}

if (-not (Test-Path -LiteralPath $backend)) {
    Write-Fail "archive layout" "rustc_codegen_cuda.dll is missing"
}

Write-Pass "archive layout"

$env:CUDA_OXIDE_BACKEND = $backend
& $cargoOxide --version
if ($LASTEXITCODE -ne 0) {
    Write-Fail "cargo-oxide --version" "exit code $LASTEXITCODE"
}
Write-Pass "cargo-oxide --version"

if (Get-Command rustc -ErrorAction SilentlyContinue) {
    rustc --version
    Write-Pass "rustc on PATH"
} else {
    Write-Skip "rustc on PATH" "rustup toolchain not visible"
}

if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version
    Write-Pass "nvcc on PATH"
} else {
    Write-Skip "nvcc on PATH" "CUDA Toolkit bin directory not visible"
}

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    cargo --version
    Write-Pass "cargo on PATH"
} elseif ($RequireCargo) {
    Write-Fail "cargo on PATH" "RequireCargo was set"
} else {
    Write-Skip "cargo on PATH" "cargo not visible"
}

Write-Host "SMOKE SUMMARY: PASS"
