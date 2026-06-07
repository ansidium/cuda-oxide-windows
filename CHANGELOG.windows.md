# Windows Support Notes

## Unreleased

### Supported Target

- `x86_64-pc-windows-msvc` on Windows 10 22H2 and Windows 11 is experimental.
- Linux remains upstream-compatible with NVlabs/cuda-oxide.
- `x86_64-pc-windows-gnu` is unsupported initially.

### Requirements

- NVIDIA GPU and a driver compatible with the installed CUDA Toolkit.
- CUDA Toolkit 12.x or 13.x.
- Visual Studio 2022 Build Tools with MSVC x64 and Windows SDK.
- Rust nightly from `rust-toolchain.toml`.
- Rust components: `rust-src`, `rustc-dev`, `llvm-tools`.
- Clang/libclang for `bindgen`.

### Validated Examples

- `vecadd` is the primary Windows readiness example.
- Build-only readiness: `cargo oxide build vecadd`.
- Runtime readiness on a GPU host: `cargo oxide run vecadd`.
- Scripted readiness: `.\scripts\smoketest.ps1`.

### Unsupported Items

- Windows GNU target.
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
