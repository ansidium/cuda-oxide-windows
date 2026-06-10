/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #129: kernel calling libdevice float math
//! (`__nv_sqrtf` / `__nv_floorf` / `__nv_fmaf`) puts the build on the
//! auto-detected NVVM IR path. Inspect `issue129_repro.ll` for:
//!   - the datalayout string used,
//!   - whether the kernel carries the `ptx_kernel` calling convention,
//!   - whether `!nvvm.annotations` marks the kernel instead.

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
    // Compile-only repro; no GPU execution in the triage sandbox.
    println!("issue129_repro: device compilation only");
}
