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
- Short-lived branches may be used for experiments, but routine upstream sync
  lands directly on `main`.

## Branch Rules

- `main` remains the Windows release fork branch. It tracks
  `NVlabs/cuda-oxide` `upstream/main`.
- Short-lived branches may be used for Windows enablement work, experiments,
  CI fixes, and compatibility patches.
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

Regular sync should use this fetch-plus-merge flow on `main` so the fork keeps
its published history intact while still carrying upstream commits promptly.

The working tree must be clean before merging. If you have unfinished Windows
work, move it to a short-lived branch before syncing.

Merge upstream:

```bash
git merge --no-ff upstream/main
```

Do not rewrite published `main` history for routine syncs.

Conflict policy:

- Preserve upstream behavior first. Prefer the upstream file when behavior is
  unrelated to Windows support.
- Re-apply Windows helpers after the upstream version is understood.
- Keep compatibility patches narrow: paths, environment discovery, CI, docs,
  and smoke support before public API changes.
- Do not use conflict resolution to introduce Linux behavior changes.
- Update this file only when the resolved result intentionally diverges from
  upstream behavior.

No-GPU sync checks after merge:

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

The helper merges `upstream/main` into `main` and, when `-Push` is passed,
uses a normal `git push origin main`. It does not create releases, move tags,
or change versions.

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

- Daily and weekly upstream monitor: `.github/workflows/upstream-monitor.yml`
  compares this fork with `NVlabs/cuda-oxide/main` and opens or updates one
  issue when upstream has new commits.
- Weekly upstream sync: `.github/workflows/upstream-sync-main.yml` merges
  `NVlabs/cuda-oxide/main` into `main`, runs `.\scripts\sync-upstream.ps1
  -RunChecks -Push`, and opens or updates one issue if the sync fails.
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

## 2026-06-22 - Upstream sync and non-rewriting maintenance

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@d63a0a8d3fef2db450ee342bdcd862a7829c3cbb`,
  CUDA-Oxide 0.2.1.
- Files/area: upstream merge integration, cargo-oxide backend cache handling,
  cuda-core upstream behavior, and sync automation.
- Intentional divergence: keep the Windows support layer on current upstream
  through merge-based `main` syncs.
- Linux impact: intended to be none. Upstream `DeviceBuffer` behavior and
  backend cache source/toolchain invalidation are preserved; Windows-specific
  artifact naming, release-profile backend builds, and loader path handling
  remain scoped to Windows targets.
- Windows validation: run the no-GPU sync sequence and hosted Windows canary
  before publishing release artifacts.
- Follow-up: no new `windows-vX.Y.Z` release is needed until upstream publishes
  a new CUDA-Oxide release baseline.

## 2026-06-18 - Upstream sync and stronger Windows canary

- Branch: `main` Windows release fork branch tracking `upstream/main`.
- Upstream baseline: `upstream/main@56b843f618d973aef6ae4cb613b590008df09a70`,
  CUDA-Oxide 0.2.1.
- Files/area: Windows CI, cargo-oxide backend loader-path handling, and
  cuda-core sync repair.
- Intentional divergence: keep the Windows support layer synced with current
  upstream while strengthening the hosted no-GPU Windows canary to install
  `libffi:x64-windows` through a temporary vcpkg manifest seeded with the
  runner's vcpkg baseline, build the codegen backend, and compile-check
  `vecadd`.
- Linux impact: intended to be none. The sync repair preserves the existing
  `DeviceBuffer` behavior while filling fields required by upstream's
  async-free model.
- Windows validation: `cargo oxide setup` built
  `rustc_codegen_cuda.dll` in release profile, and
  `cargo oxide build vecadd --arch sm_75` completed successfully.
- Follow-up: no new `windows-vX.Y.Z` release is needed until upstream publishes
  a new CUDA-Oxide release baseline.

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
