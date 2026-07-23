# Windows Support Notes

## Unreleased

- Switched the workspace, devcontainer, CLI scaffold, CI, and documentation
  from a dated nightly pin to the latest stable Rust channel.
- Adapted the MIR importer and private codegen-backend API to stable Rust 1.97.
- Merged `main` with
  `upstream/main@8d02eac17e56d5ed5ad3713fc498a2591d810919`.
- Added a weekly upstream sync workflow for `main`; failed syncs open or update
  one issue.
- Strengthened the hosted Windows no-GPU canary so the regular `windows`
  workflow installs `libffi:x64-windows` through a temporary vcpkg manifest,
  seeds that manifest with the runner's vcpkg baseline, builds the codegen
  backend, and compile-checks `vecadd` with `cargo oxide build vecadd --arch sm_75`.
- Restored `DeviceBuffer` context binding, empty-buffer copy fast paths, and
  the previous `copy_from_host_async(src, stream)` entry point after merging
  upstream's safer host-copy API.
- Resolved upstream-sync fallout around `DeviceBuffer` field initialization
  after upstream's async-free changes.
- No new Windows release was published; the upstream release baseline remains
  CUDA-Oxide `v0.2.1`.

## windows-v0.2.1 - 2026-06-14

### Upstream Baseline and Versioning

- Current upstream baseline:
  `upstream/main@cb318ad4e4e37f5e1913ed0a13478af990e857f7`.
- Upstream release baseline: CUDA-Oxide `v0.2.1`.
- Project version remains CUDA-Oxide 0.2.1.
- Publish `windows-vX.Y.Z` only when upstream has released `vX.Y.Z`;
  sync-only rebuilds do not invent new project versions.

### Maintenance Automation

- Added a daily and weekly upstream monitor workflow for `NVlabs/cuda-oxide`
  drift.
- Added a weekly hosted Windows no-GPU canary for the MSVC target.
- Kept release publishing manual so tags, artifacts, and signatures stay under
  maintainer control.
- Build the codegen backend in release profile on Windows to avoid MSVC debug
  linker object-count limits.

### Supported Target

- Windows support is scoped to `x86_64-pc-windows-msvc` on Windows 10 22H2
  and Windows 11.
- Linux remains upstream-compatible with NVlabs/cuda-oxide.

### Requirements

- NVIDIA GPU and a driver compatible with the installed CUDA Toolkit.
- CUDA Toolkit 12.x or 13.x.
- Visual Studio 2022 Build Tools with MSVC x64 and Windows SDK.
- Stable Rust from `rust-toolchain.toml`.
- Rust components: `rust-src`, `rustc-dev`, `llvm-tools`.
- Clang/libclang for `bindgen`.

### Validated Examples

- `vecadd` is the primary Windows readiness example.
- Build-only readiness: `cargo oxide build vecadd`.
- Runtime readiness on a GPU host: `cargo oxide run vecadd`.
- Scripted readiness: `.\scripts\smoketest.ps1`.

### Unsupported Items

- CUDA examples that require Linux-only tooling such as `cuda-gdb`.
- Full GPU smoke on GitHub-hosted `windows-latest` runners, because they do not
  provide NVIDIA GPU hardware.

### Release-Readiness Checklist

- `cargo fmt --all --check`
- `cargo build -p cargo-oxide`
- `cargo test -p oxide-artifacts --features object`
- `cargo oxide doctor`
- `cargo oxide build vecadd`
- `cargo oxide run vecadd` on a Windows GPU host
