param(
    [string]$UpstreamRemote = 'upstream',
    [string]$UpstreamBranch = 'main',
    [string]$LocalBranch = 'main',
    [switch]$RunChecks,
    [switch]$Push
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-ToolAvailable {
    param(
        [string]$Name
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH."
    }
}

function Invoke-ExternalCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    Write-Host "RUN: $Name"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

function Invoke-ExternalOutput {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $output = & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }

    return ($output -join "`n").Trim()
}

function Invoke-Git {
    param(
        [string]$Name,
        [string[]]$ArgumentList
    )

    Invoke-ExternalCommand -Name $Name -FilePath 'git' -ArgumentList $ArgumentList
}

function Invoke-GitOutput {
    param(
        [string]$Name,
        [string[]]$ArgumentList
    )

    return Invoke-ExternalOutput -Name $Name -FilePath 'git' -ArgumentList $ArgumentList
}

function Assert-NoGitOperationInProgress {
    $paths = @(
        [pscustomobject]@{ Name = 'rebase'; Path = Invoke-GitOutput 'git rev-parse --git-path rebase-merge' @('rev-parse', '--git-path', 'rebase-merge') },
        [pscustomobject]@{ Name = 'rebase'; Path = Invoke-GitOutput 'git rev-parse --git-path rebase-apply' @('rev-parse', '--git-path', 'rebase-apply') },
        [pscustomobject]@{ Name = 'merge'; Path = Invoke-GitOutput 'git rev-parse --git-path MERGE_HEAD' @('rev-parse', '--git-path', 'MERGE_HEAD') },
        [pscustomobject]@{ Name = 'cherry-pick'; Path = Invoke-GitOutput 'git rev-parse --git-path CHERRY_PICK_HEAD' @('rev-parse', '--git-path', 'CHERRY_PICK_HEAD') }
    )

    foreach ($item in $paths) {
        if (Test-Path -LiteralPath $item.Path) {
            throw "Refusing to run because a $($item.Name) is already in progress."
        }
    }
}

function Assert-CleanWorkingTree {
    $status = Invoke-GitOutput 'git status --porcelain' @('status', '--porcelain=v1', '--untracked-files=all')
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        throw "Refusing to rebase because the working tree is dirty."
    }
}

function Assert-RemoteExists {
    param(
        [string]$Name
    )

    $remotes = @((Invoke-GitOutput 'git remote' @('remote')) -split "`n")
    if ($remotes -notcontains $Name) {
        throw "Remote '$Name' is missing."
    }
}

function Assert-SupportedLocalBranch {
    param(
        [string]$Branch
    )

    if ($Branch -ne 'main') {
        throw "Refusing to sync branch '$Branch'; this helper is locked to the Windows fork main branch."
    }
}

function Assert-SupportedUpstreamTarget {
    param(
        [string]$Remote,
        [string]$Branch
    )

    if ($Remote -ne 'upstream' -or $Branch -ne 'main') {
        throw "Refusing to sync from '$Remote/$Branch'; this helper is locked to upstream/main."
    }
}

function Assert-CurrentBranch {
    param(
        [string]$ExpectedBranch
    )

    $currentBranch = Invoke-GitOutput 'git branch --show-current' @('branch', '--show-current')
    if ($currentBranch -ne $ExpectedBranch) {
        throw "Refusing to run on branch '$currentBranch'; expected '$ExpectedBranch'."
    }
}

function Invoke-Checks {
    $checks = @(
        [pscustomobject]@{ Name = 'cargo fmt --all --check'; Args = @('fmt', '--all', '--check') },
        [pscustomobject]@{ Name = 'cargo test -p cargo-oxide'; Args = @('test', '-p', 'cargo-oxide') },
        [pscustomobject]@{ Name = 'cargo test -p cuda-core --lib'; Args = @('test', '-p', 'cuda-core', '--lib') },
        [pscustomobject]@{ Name = 'cargo test -p cuda-async'; Args = @('test', '-p', 'cuda-async') },
        [pscustomobject]@{ Name = 'cargo test -p cuda-host'; Args = @('test', '-p', 'cuda-host') },
        [pscustomobject]@{ Name = 'cargo test -p oxide-artifacts --features object'; Args = @('test', '-p', 'oxide-artifacts', '--features', 'object') },
        [pscustomobject]@{ Name = 'cargo clippy --workspace -- -D warnings'; Args = @('clippy', '--workspace', '--', '-D', 'warnings') },
        [pscustomobject]@{ Name = 'cargo doc --no-deps --workspace'; Args = @('doc', '--no-deps', '--workspace') }
    )

    foreach ($check in $checks) {
        Invoke-ExternalCommand -Name $check.Name -FilePath 'cargo' -ArgumentList $check.Args
    }
}

function Write-SyncStatus {
    param(
        [string]$Name,
        [string]$Value
    )

    Write-Host "${Name}: $Value"
}

Assert-ToolAvailable 'git'

try {
    $isWorkTree = Invoke-GitOutput 'git rev-parse --is-inside-work-tree' @('rev-parse', '--is-inside-work-tree')
} catch {
    throw "Current directory is not a git worktree."
}

if ($isWorkTree -ne 'true') {
    throw "Current directory is not a git worktree."
}

$repoRoot = Invoke-GitOutput 'git rev-parse --show-toplevel' @('rev-parse', '--show-toplevel')
Set-Location $repoRoot

Assert-NoGitOperationInProgress
Assert-CleanWorkingTree
Assert-SupportedLocalBranch $LocalBranch
Assert-SupportedUpstreamTarget $UpstreamRemote $UpstreamBranch
Assert-RemoteExists $UpstreamRemote
if ($Push) {
    Assert-RemoteExists 'origin'
}
Assert-CurrentBranch $LocalBranch

$oldHead = Invoke-GitOutput 'git rev-parse HEAD' @('rev-parse', 'HEAD')
Write-SyncStatus 'Old HEAD' $oldHead

Invoke-Git "git fetch $UpstreamRemote --tags" @('fetch', $UpstreamRemote, '--tags')

$upstreamRef = "$UpstreamRemote/$UpstreamBranch"
$upstreamHead = ''
try {
    $upstreamHead = Invoke-GitOutput "git rev-parse --verify $upstreamRef" @('rev-parse', '--verify', "${upstreamRef}^{commit}")
} catch {
    throw "Upstream ref '$upstreamRef' could not be resolved after fetch."
}
Write-SyncStatus 'Upstream HEAD' $upstreamHead

Invoke-Git "git rebase $upstreamRef" @('rebase', $upstreamRef)

$newHead = Invoke-GitOutput 'git rev-parse HEAD' @('rev-parse', 'HEAD')
Write-SyncStatus 'New HEAD' $newHead

if ($RunChecks) {
    try {
        Invoke-Checks
        Write-SyncStatus 'Checks' 'passed'
    } catch {
        Write-SyncStatus 'Checks' 'failed'
        Write-SyncStatus 'Push' 'skipped (checks failed)'
        throw
    }
} else {
    Write-SyncStatus 'Checks' 'skipped (-RunChecks not set)'
}

if ($Push) {
    try {
        Invoke-Git 'git push --force-with-lease origin main' @('push', '--force-with-lease', 'origin', 'main')
        Write-SyncStatus 'Push' 'pushed origin main with --force-with-lease'
    } catch {
        Write-SyncStatus 'Push' 'failed'
        throw
    }
} else {
    Write-SyncStatus 'Push' 'skipped (-Push not set)'
}
