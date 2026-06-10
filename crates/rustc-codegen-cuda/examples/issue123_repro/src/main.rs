/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #123: float `!=` must lower to `fcmp une` (unordered),
//! not `fcmp one` (ordered). Rust semantics: `NaN != NaN` is `true`, so
//! the canonical `x != x` NaN check must survive lowering.
//!
//! Compile-only check (no GPU needed):
//!   cargo oxide build issue123_repro --arch sm_90 -v
//!   grep fcmp crates/rustc-codegen-cuda/examples/issue123_repro/issue123_repro.ll

use cuda_device::{DisjointSlice, kernel, thread};

/// Canonical NaN check: `v != v` is true iff v is NaN.
/// Correct LLVM IR: `fcmp une float %v, %v`.
/// Buggy LLVM IR:   `fcmp one float %v, %v` (always false; opt folds it away).
#[kernel]
pub fn is_nan(x: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(ce) = c.get_mut(idx) {
        let v = x[i];
        *ce = if v != v { 1.0 } else { 0.0 };
    }
}

/// General float `!=` between two distinct values, so the comparison cannot
/// be folded even when `x != x` is canonicalised: NaN on either side must
/// make this true under Rust semantics.
#[kernel]
pub fn float_ne(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(ce) = c.get_mut(idx) {
        *ce = if a[i] != b[i] { 1.0 } else { 0.0 };
    }
}

fn main() {
    // Compile-only repro; device IR inspection happens on the generated .ll.
    println!("issue123_repro: inspect issue123_repro.ll for the fcmp predicate");
}
