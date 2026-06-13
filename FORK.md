# cuda-oxide Windows Fork Policy

## Purpose

This fork carries experimental native Windows support for cuda-oxide while
keeping Linux behavior upstream-compatible. Windows support is scoped to
developer enablement, CI canaries, smoke scripts, and narrow compatibility
helpers needed for `x86_64-pc-windows-msvc`.

## Upstream Repository

- Upstream: https://github.com/NVlabs/cuda-oxide
- Upstream branch: `NVlabs/cuda-oxide` `upstream/main`
- Upstream release baseline: CUDA-Oxide 0.2.1
- Primary local branch: `main` Windows release fork branch tracking
  `upstream/main`
- Windows branches: `custom/windows-*`

## Branch Rules

- `main` remains the Windows release fork branch. It tracks
  `NVlabs/cuda-oxide` `upstream/main` and should stay suitable for Linux users
  following NVlabs/cuda-oxide documentation.
- `custom/windows-*` branches are used for Windows enablement work, experiments,
  CI fixes, and short-lived compatibility patches.
- Keep Windows patches narrow and reviewable. Prefer build-system, path,
  environment, and documentation helpers over API changes.
- Do not add fork-only public API unless it has been discussed and documented.
- Linux regressions block Windows changes unless the pull request documents why
  the regression is unavoidable and what follow-up restores parity.
- Update the divergence log only for intentional behavior differences, not for
  transient merge conflict resolution or ordinary documentation refreshes.

## Upstream Sync

Configure the upstream remote once:

```bash
git remote add upstream https://github.com/NVlabs/cuda-oxide.git
git remote -v
```

Refresh upstream state:

```bash
git fetch upstream --tags
git checkout main
git status --short
```

Regular sync should use this fetch-plus-rebase flow on `main` so upstream
history remains easy to inspect and Windows fork patches stay reviewable.

The working tree must be clean before rebasing. If you have unfinished Windows
work, move it to a `custom/windows-*` branch before syncing.

Rebase onto upstream:

```bash
git rebase upstream/main
```

If local integration policy requires a merge commit instead, keep the same
conflict policy and record the reason in the integration notes.

Conflict policy:

- Preserve upstream behavior first. Prefer the upstream file when behavior is
  unrelated to Windows support.
- Re-apply Windows helpers after the upstream version is understood.
- Keep compatibility patches narrow: paths, environment discovery, CI, docs,
  and smoke support before public API changes.
- Do not use conflict resolution to introduce Linux behavior changes.
- Update this file only when the resolved result intentionally diverges from
  upstream behavior.

No-GPU sync checks after rebase:

```powershell
cargo fmt --all --check
cargo test -p cargo-oxide
cargo test -p cuda-core --lib
cargo test -p cuda-async
cargo test -p cuda-host
cargo test -p oxide-artifacts --features object
cargo clippy --workspace -- -D warnings
cargo doc --no-deps --workspace
```

Run the canonical no-GPU sequence with:

```powershell
.\scripts\sync-upstream.ps1 -RunChecks
```

Push a completed sync only after the checks pass:

```powershell
.\scripts\sync-upstream.ps1 -RunChecks -Push
```

The helper rebases `main` onto `upstream/main` and, when `-Push` is passed,
uses `git push --force-with-lease`. It does not create releases, move tags, or
change versions.

On a Windows GPU host, also run the full Windows smoke path:

```powershell
cargo oxide doctor
cargo oxide build vecadd
cargo oxide run vecadd
.\scripts\smoketest.ps1
```

When a sync changes Windows readiness, update
[CHANGELOG.windows.md](CHANGELOG.windows.md). Keep the changelog focused on
validated targets, requirements, examples, unsupported items, and known
release-readiness gaps.

## Maintenance Cycle

- Daily upstream monitor: `.github/workflows/upstream-monitor.yml` compares this
  fork with `NVlabs/cuda-oxide/main` and opens or updates one maintenance issue
  when upstream has new commits.
- Weekly upstream monitor: the same workflow runs a weekly cadence for a
  slower, human-friendly sync checkpoint.
- Weekly hosted Windows canary: `.github/workflows/windows.yml` runs the
  no-GPU MSVC lane on GitHub-hosted `windows-latest`.
- Manual sync: run `.\scripts\sync-upstream.ps1 -RunChecks`; add `-Push` only
  after the local result is clean.
- Release rule: publish `windows-vX.Y.Z` only when upstream has released
  `vX.Y.Z`. Sync-only rebuilds do not invent new project versions.

## Divergence Log Format

Use this format whenever the fork intentionally keeps behavior that differs
from upstream:

```text
## YYYY-MM-DD - short title

- Branch:
- Upstream baseline:
- Files/area:
- Intentional divergence:
- Linux impact:
- Windows validation:
- Follow-up:
```

## Current Divergence Log

## 2026-06-14 - Windows support layer

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@cb318ad4e4e37f5e1913ed0a13478af990e857f7`,
  CUDA-Oxide 0.2.1.
- Files/area: CUDA Toolkit discovery, Windows/MSVC path handling, platform
  artifact naming, loader environment handling, import-library checks,
  bindgen/CUDA compatibility, Windows CI, and smoke scripts.
- Intentional divergence: add native Windows support infrastructure for
  `x86_64-pc-windows-msvc` while keeping upstream Linux behavior first. The
  Windows layer covers CUDA Toolkit discovery, Windows and MSVC paths,
  `.exe`/`.dll`/`.obj` platform naming, `PATH` loader behavior, `cuda.lib` and
  `ffi.lib` checks, bindgen/CUDA type compatibility, Windows CI, and
  PowerShell smoke scripts.
- Linux impact: intended to be none. If a Windows helper conflicts with
  upstream Linux behavior, preserve upstream behavior first and re-apply only
  the Windows-specific compatibility needed for MSVC support.
- Windows validation: `cargo build -p cargo-oxide`,
  `cargo test -p oxide-artifacts --features object`,
  `cargo oxide doctor`, `cargo oxide build vecadd`, GPU-host
  `cargo oxide run vecadd`, and `.\scripts\smoketest.ps1` as applicable.
- Follow-up: during each upstream sync, check whether upstream has added native
  equivalents and remove fork-only helpers when upstream behavior covers the
  Windows case.
