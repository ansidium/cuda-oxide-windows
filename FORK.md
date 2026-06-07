# cuda-oxide Windows Fork Policy

## Purpose

This fork carries experimental native Windows support for cuda-oxide while
keeping Linux behavior upstream-compatible. Windows support is scoped to
developer enablement, CI canaries, smoke scripts, and narrow compatibility
helpers needed for `x86_64-pc-windows-msvc`.

## Upstream Repository

- Upstream: https://github.com/NVlabs/cuda-oxide
- Upstream baseline: CUDA-Oxide 0.2.0
- Primary local branch: `main`
- Windows branches: `custom/windows-*`

## Branch Rules

- `main` tracks upstream-facing behavior and should stay suitable for Linux
  users following NVlabs/cuda-oxide documentation.
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

The working tree must be clean before rebasing. If you have unfinished Windows
work, move it to a `custom/windows-*` branch before syncing.

Rebase onto upstream:

```bash
git rebase upstream/main
```

If local integration policy requires a merge commit instead, keep the same
conflict policy and record the reason in the integration notes.

Conflict policy:

- Preserve upstream first. Prefer the upstream file when behavior is unrelated
  to Windows support.
- Re-apply Windows helpers after the upstream version is understood.
- Keep compatibility patches narrow: paths, environment discovery, CI, docs,
  and smoke support before public API changes.
- Do not use conflict resolution to introduce Linux behavior changes.
- Update this file only when the resolved result intentionally diverges from
  upstream behavior.

Linux checks after sync:

```bash
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
cargo test -p oxide-artifacts --features object
cargo test -p cargo-oxide
```

Windows MSVC checks after sync:

```powershell
cargo fmt --all --check
cargo build -p cargo-oxide
cargo test -p oxide-artifacts --features object
.\scripts\smoketest.ps1 -BuildOnly
```

On a Windows GPU host, also run:

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

No intentional code-level divergence is recorded in this document yet. The
current fork-specific surface is documentation, CI, release notes, and smoke
script support.
