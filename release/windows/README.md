# cuda-oxide Windows Release

This archive contains the Windows x86_64 MSVC build of the cuda-oxide cargo
subcommand and codegen backend.

## Files

- `cargo-oxide.exe` - the `cargo oxide` subcommand binary.
- `rustc_codegen_cuda.dll` - the rustc codegen backend.
- `README.md` - this release note.
- `smoketest.ps1` - a lightweight archive sanity check.

Keep `cargo-oxide.exe` and `rustc_codegen_cuda.dll` in the same directory. The
CLI checks this side-by-side layout before falling back to workspace, cache, or
auto-fetch discovery.

## Target

`x86_64-pc-windows-msvc`

## Requirements

- Windows 10/11 x86_64.
- Rustup with the stable toolchain selected by the cuda-oxide repository.
- CUDA Toolkit 12.x or 13.x, including `nvcc`, `cuda.h`, `cuda.lib`,
  `nvvm64_*.dll`, `nvJitLink_*.dll`, and `libdevice.10.bc`.
- Visual Studio 2022 Build Tools with MSVC x64 and a Windows SDK.
- LLVM/Clang with libclang available on `PATH` or through `LIBCLANG_PATH`.

## Quick Check

From the extracted archive:

```powershell
.\smoketest.ps1
```

For a full environment check, run `cargo oxide doctor` from a cuda-oxide
checkout or a cuda-oxide project. The side-by-side backend DLL is discovered
automatically from this archive layout.

The archive does not include CUDA Toolkit DLLs, rustup toolchain DLLs, LLVM, or
MSVC runtime redistributables. Install those through their normal installers.

This is an unofficial Windows-support fork release, not an official NVIDIA Labs
release.
