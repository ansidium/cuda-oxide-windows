param(
    [string]$Version = "",
    [ValidateSet("release", "debug")]
    [string]$Profile = "release",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$OutDir = "",
    [string]$SignScript = "",
    [switch]$SkipBuild,
    [switch]$SkipSign
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

if ($TargetTriple -ne "x86_64-pc-windows-msvc") {
    throw "Only x86_64-pc-windows-msvc release archives are supported."
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "target\windows-release"
}

function Invoke-Checked {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory = $repoRoot
    )

    Write-Host "RUN: $Name"
    Push-Location $WorkingDirectory
    try {
        & $Exe @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Add-PathValue {
    param(
        [string]$Name,
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $existing = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($existing)) {
        [Environment]::SetEnvironmentVariable($Name, $Path, "Process")
    } else {
        [Environment]::SetEnvironmentVariable($Name, "$Path;$existing", "Process")
    }
}

function Get-PackageVersion {
    Push-Location $repoRoot
    try {
        $metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
        $pkg = $metadata.packages | Where-Object { $_.name -eq "cargo-oxide" } | Select-Object -First 1
        if (-not $pkg) {
            throw "cargo-oxide package was not found in cargo metadata"
        }
        return $pkg.version
    } finally {
        Pop-Location
    }
}

function Find-Libffi {
    $candidates = @()

    if ($env:LIBFFI_LIB_DIR) {
        $candidates += [pscustomobject]@{
            LibDir = $env:LIBFFI_LIB_DIR
            BinDir = $env:LIBFFI_BIN_DIR
        }
    }

    if ($env:LIB) {
        foreach ($libDir in ($env:LIB -split ";")) {
            if (-not [string]::IsNullOrWhiteSpace($libDir)) {
                $candidates += [pscustomobject]@{
                    LibDir = $libDir
                    BinDir = ""
                }
            }
        }
    }

    $roots = @()
    if ($env:VCPKG_ROOT) {
        $roots += $env:VCPKG_ROOT
    }

    $vcpkg = Get-Command vcpkg -ErrorAction SilentlyContinue
    if ($vcpkg) {
        $roots += Split-Path -Parent $vcpkg.Source
    }

    foreach ($root in $roots | Select-Object -Unique) {
        $installed = Join-Path $root "installed\x64-windows"
        $candidates += [pscustomobject]@{
            LibDir = Join-Path $installed "lib"
            BinDir = Join-Path $installed "bin"
        }
    }

    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate.LibDir)) {
            $lib = Join-Path $candidate.LibDir "ffi.lib"
            if (Test-Path -LiteralPath $lib) {
                return $candidate
            }
        }
    }

    return $null
}

function Enable-LibffiForBuild {
    $libffi = Find-Libffi
    if (-not $libffi) {
        throw "ffi.lib was not found. Install libffi:x64-windows with vcpkg or set LIBFFI_LIB_DIR."
    }

    Add-PathValue "LIB" $libffi.LibDir
    if (-not [string]::IsNullOrWhiteSpace($libffi.BinDir)) {
        Add-PathValue "PATH" $libffi.BinDir
    }

    Write-Host "Using libffi import library: $(Join-Path $libffi.LibDir 'ffi.lib')"
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = Get-PackageVersion
}

Enable-LibffiForBuild

$profileArgs = @()
if ($Profile -eq "release") {
    $profileArgs += "--release"
}

if (-not $SkipBuild) {
    Invoke-Checked "cargo build -p cargo-oxide $Profile" "cargo" (@("build", "-p", "cargo-oxide") + $profileArgs)
    Invoke-Checked "cargo build rustc_codegen_cuda $Profile" "cargo" (@("build") + $profileArgs) (Join-Path $repoRoot "crates\rustc-codegen-cuda")
}

$profileDir = $Profile
$cargoOxide = Join-Path $repoRoot "target\$profileDir\cargo-oxide.exe"
$backend = Join-Path $repoRoot "crates\rustc-codegen-cuda\target\$profileDir\rustc_codegen_cuda.dll"

if (-not (Test-Path -LiteralPath $cargoOxide)) {
    throw "Missing artifact: $cargoOxide"
}
if (-not (Test-Path -LiteralPath $backend)) {
    throw "Missing artifact: $backend"
}

if (-not $SkipSign) {
    if ([string]::IsNullOrWhiteSpace($SignScript)) {
        Write-Host "Skipping signing; pass -SignScript <path> to sign release binaries."
    } else {
        if (-not (Test-Path -LiteralPath $SignScript)) {
            throw "Signing script was not found: $SignScript"
        }
        Invoke-Checked "sign release binaries" $SignScript @($cargoOxide, $backend)
    }
}

$archiveBase = "cuda-oxide-v$Version-$TargetTriple"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$packageDir = Join-Path $OutDir $archiveBase

if (Test-Path -LiteralPath $packageDir) {
    $resolvedOut = (Resolve-Path -LiteralPath $OutDir).Path
    $resolvedPackage = (Resolve-Path -LiteralPath $packageDir).Path
    if (-not $resolvedPackage.StartsWith($resolvedOut, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove package directory outside OutDir: $resolvedPackage"
    }
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

Copy-Item -LiteralPath $cargoOxide -Destination (Join-Path $packageDir "cargo-oxide.exe")
Copy-Item -LiteralPath $backend -Destination (Join-Path $packageDir "rustc_codegen_cuda.dll")
Copy-Item -LiteralPath (Join-Path $repoRoot "release\windows\README.md") -Destination (Join-Path $packageDir "README.md")
Copy-Item -LiteralPath (Join-Path $repoRoot "release\windows\smoketest.ps1") -Destination (Join-Path $packageDir "smoketest.ps1")

$zipPath = Join-Path $OutDir "$archiveBase.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $zipPath -Force
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
"$zipHash  $(Split-Path -Leaf $zipPath)" | Set-Content -LiteralPath "$zipPath.sha256" -Encoding ASCII

Write-Host "Package directory: $packageDir"
Write-Host "Release archive: $zipPath"
Write-Host "Release SHA256: $zipHash"
