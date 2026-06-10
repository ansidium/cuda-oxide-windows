/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #132: `*None::<&u32>.unwrap_or(&77)` faults with CUDA 700.
//!
//! With -O, MIR const-folds the `None` arm's `unwrap_or(&77)` into a
//! reference-to-scalar constant (`ConstantKind::Allocated`, 8 zero pointer
//! bytes plus a provenance entry pointing at the `77` allocation). The
//! importer's pointer-constant arm follows that provenance only when the
//! pointee is a struct; for a scalar pointee it falls through to the raw
//! pointer path and emits `inttoptr 0` (a null pointer). The dereference
//! then becomes `load i32, ptr null` -> illegal memory access at runtime.
//!
//! Compile-only check (no GPU needed):
//!   cargo oxide build issue132_repro --arch sm_90 -v
//!   grep -n 'inttoptr\|load i32' issue132_repro.ll
//!
//! Correct IR must materialize the 77 (e.g. as a stack/global value whose
//! address is taken); buggy IR loads through a null pointer.

use cuda_device::{kernel, thread};

#[kernel]
pub fn opt_ref_unwrap_or(out: &[u32]) {
    if thread::index_1d().get() != 0 {
        return;
    }
    let r: u32 = 5;
    let a: Option<&u32> = Some(&r);
    let b: Option<&u32> = None; // keeping BOTH a Some and a None live triggers
    // the const-fold of the None arm's unwrap_or
    let v0: u32 = *a.unwrap_or(&77); // 5
    let v1: u32 = *b.unwrap_or(&77); // should be 77

    unsafe {
        let p = out.as_ptr() as *mut u32;
        *p.add(0) = v0;
        *p.add(1) = v1;
    }
}

fn main() {
    // Compile-only repro; device IR inspection happens on the generated .ll.
    println!("issue132_repro: inspect issue132_repro.ll for `load` through a null pointer");
}
