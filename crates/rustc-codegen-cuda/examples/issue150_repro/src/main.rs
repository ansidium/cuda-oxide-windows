/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #150: "PTX generation failed; Intrinsic has incorrect
//! argument type! ptr @llvm.lifetime.start.p0"
//!
//! Kernel taken verbatim from the issue report. The inlined `check3` body
//! (gpu_printf arg buffer allocas) makes `opt -O2`'s inliner insert
//! llvm.lifetime.start/end markers. LLVM 22 dropped the `i64` size
//! parameter from those intrinsics, so IR optimised by an LLVM 22 `opt`
//! fails an LLVM 21 `llc`'s verifier.
//!
//! Before the fix, the pipeline discovered `opt` independently of `llc`,
//! so pinning `CUDA_OXIDE_LLC` to an LLVM 21 `llc` while the rustc
//! sysroot ships LLVM 22 llvm-tools reproduced the failure:
//!
//!   CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide build issue150_repro --arch sm_90
//!
//! With matched-pair toolchain resolution the same command succeeds: the
//! pipeline picks an LLVM 21 `opt` to go with the pinned `llc` (or skips
//! the middle-end with a warning when none exists).

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, gpu_printf, kernel, thread};

    // Kernel shape kept verbatim from issue #150: this exact pattern is
    // what makes opt's inliner introduce llvm.lifetime.start/end markers.
    #[allow(clippy::redundant_pattern_matching)]
    #[kernel]
    pub fn misbehave(mut dst: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(_) = dst.get_mut(idx) {
            check3();
        }
    }

    fn check3() {
        {
            let (r, borrow) = 0x4bf9_0000u32.overflowing_sub(0xf329_0000);
            gpu_printf!("{:x}, {}\n", r, borrow);
        }
    }
}

fn main() {
    // No GPU needed: this repro only exercises device compilation. The
    // SUCCESS marker satisfies the smoketest standard-category pass rule.
    println!("issue150_repro: device compile-only repro - SUCCESS");
}
