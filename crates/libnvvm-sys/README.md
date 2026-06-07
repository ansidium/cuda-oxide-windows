# libnvvm-sys

Runtime (`dlopen`) bindings to NVIDIA's libNVVM. libNVVM is the front-end of NVIDIA's PTX-targeting compiler; it accepts NVVM IR (an LLVM-IR dialect) and produces PTX or LTOIR.

## What this crate provides

- `LibNvvm` — RAII wrapper around the loaded library + resolved function pointers.
- `Program` — RAII wrapper around an `nvvmProgram` handle, with `add_module` / `compile` methods.
- `NvvmError` — typed errors with the libNVVM error log captured.

## Build requirements

None. The library is loaded at runtime, so the CUDA Toolkit only needs to be present when the program runs (not when it compiles).

## Library discovery

`LibNvvm::load()` tries (in order):

1. `LIBNVVM_PATH` env var, if set.
2. Platform loader candidates:
   - Linux: `libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so`.
   - Windows: discovered `nvvm64_*.dll` files on the normal DLL search path.
3. CUDA Toolkit roots from the shared toolkit discovery helper:
   - Linux: `<root>/nvvm/lib64/libnvvm.so`.
   - Windows: `<root>\nvvm\bin\x64\nvvm64_*.dll`.

libNVVM ships with the standard CUDA Toolkit. On Linux it is under `<cuda>/nvvm/lib64/`; on Windows it is typically a versioned DLL such as `nvvm64_40_0.dll` under `<cuda>\nvvm\bin\x64\`. Set `LIBNVVM_PATH` to a full library path when you need an explicit override.

## Usage

This crate is low-level. Most users want the higher-level `cuda_host::ltoir::load_kernel_module` helper, which combines libNVVM + libdevice + nvJitLink behind one call. Use this crate directly only if you need explicit control over the libNVVM compile.

```rust
use libnvvm_sys::{LibNvvm, Program};

let nvvm = LibNvvm::load()?;
let mut program = Program::new(&nvvm)?;
program.add_module(&libdevice_bytes, "libdevice.10.bc")?;
program.add_module(&kernel_ll_bytes, "kernel.ll")?;
let ltoir = program.compile(&["-arch=compute_120", "-gen-lto"])?;
```

## Companion crate

[`nvjitlink-sys`](../nvjitlink-sys/) — same pattern, for nvJitLink. Together they cover the NVVM IR → LTOIR → cubin pipeline.
