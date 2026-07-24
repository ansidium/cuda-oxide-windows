# math_hyperbolic

Exercises the hyperbolic + extended `f32`/`f64` math methods in a kernel and
checks the GPU results against host libm:

`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp_m1`, `ln_1p`, `hypot`.

## Why this exists

These methods may lower to `std::sys::cmath::*` shims depending on the Rust
toolchain. Without interception a kernel calling one of those shims fails with:

```text
CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE
Device code calls: std::sys::cmath::sinh
```

The float-math dispatch maps each supported shim to the matching libdevice
call (`__nv_sinh`, `__nv_asinh`, `__nv_hypot`, ...). `hypot` is the one
binary function. `acosh` (needs arg >= 1) and `atanh` (needs |arg| < 1) are
fed in-domain transforms of the input.

## Run

```bash
cargo oxide run math_hyperbolic
```

Exits 0 on `SUCCESS`, 1 on `FAILED`.
