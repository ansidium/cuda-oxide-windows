//! Independent repro #2 for issue #129.
//!
//! The reporter's kernel, verbatim: libdevice float math (`f32::sqrt`,
//! `f32::floor`, `f32::mul_add` lowering to `__nv_sqrtf` / `__nv_floorf`
//! / `__nv_fmaf`) forces the auto-detected NVVM-IR path. Inspect
//! `issue129_repro2.ll` for:
//!   - which datalayout string the module carries,
//!   - whether the kernel definition has the `ptx_kernel` calling
//!     convention or is instead marked via `!nvvm.annotations`,
//!   - presence of `!nvvmir.version` / `@llvm.used`.

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn fmath(input: &[f32], mut out: DisjointSlice<u32>) {
    if thread::index_1d().get() == 0 {
        let x = input[0];
        unsafe {
            *out.get_unchecked_mut(0) = x.sqrt().to_bits(); // __nv_sqrtf
            *out.get_unchecked_mut(1) = x.floor().to_bits(); // __nv_floorf
            *out.get_unchecked_mut(2) = x.mul_add(input[1], input[2]).to_bits(); // __nv_fmaf
        }
    }
}

fn main() {
    // Compile-only repro; no GPU in this environment.
    println!("issue129_repro2: device compilation only");
}
