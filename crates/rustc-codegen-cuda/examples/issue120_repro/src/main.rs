/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #120: `Rvalue::Ref` lowering drops runtime `Index`
//! projections in places shaped like `&(*ptr).field[i]`.
//!
//! Rust semantics: `&p.0[k]` (with `p: &Pair`) must produce a pointer to
//! element `k` of the inner array. The buggy lowering returns a pointer to
//! the array itself (element 0), so every read through the reference reads
//! slot 0 regardless of `k`.
//!
//! Compile-only check (no GPU needed):
//!   cargo oxide build issue120_repro --arch sm_90 -v
//!   Inspect issue120_repro.ll: the runtime index must reach the GEP that
//!   feeds the load. If it is only used by the bounds check, the bug fired.

use cuda_device::{DisjointSlice, kernel, thread};

/// Newtype wrapper so the indexed access carries a Field projection before
/// the Index projection: place = (*p).0[k] = [Deref, Field(0), Index(k)].
#[derive(Copy, Clone)]
pub struct Pair(pub [f32; 2]);

/// Out-of-line accessor: its own MIR contains
///   _0 = &(((*_1).0)[_2])
/// i.e. Rvalue::Ref of a place with projections [Deref, Field(0), Index].
/// `#[inline(never)]` keeps the shape from being dissolved by MIR opts
/// before the importer sees it.
#[inline(never)]
fn node(p: &Pair, k: usize) -> &f32 {
    &p.0[k]
}

/// Kernel 1: reference obtained through the out-of-line accessor.
/// Expected: out[i] = input[1] - input[0] (non-zero for distinct inputs).
/// Bug: both calls read slot 0, out[i] = 0.
#[kernel]
pub fn pick_via_helper(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        let pair = Pair([input[0], input[1]]);
        let r1 = *node(&pair, 1);
        let r0 = *node(&pair, 0);
        *slot = r1 - r0;
    }
}

/// Kernel 2: the same shape inline in the kernel via an explicit reborrow.
/// `pr: &Pair`, so `&pr.0[k]` is `&(*pr).0[k]` with runtime `k`.
#[kernel]
pub fn pick_direct(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        let pair = Pair([input[0], input[1]]);
        let pr: &Pair = &pair;
        // Runtime index (derived from input so it cannot const-fold).
        let k = (input[2] as usize) & 1;
        let r: &f32 = &pr.0[k];
        let _ = i;
        *slot = *r;
    }
}

/// Kernel 3: PR #121's closure shape: a move closure captures `pair`;
/// inside, `&pair.0[k]` is `&(*env).captured.0[k]`.
#[kernel]
pub fn pick_closure(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        let pair = Pair([input[0], input[1]]);
        let compute = move |k: usize| -> f32 {
            let r = &pair.0[k];
            *r
        };
        let r0 = compute(0);
        let r1 = compute(1);
        *slot = r1 - r0;
    }
}

fn main() {
    // Compile-only repro; device IR inspection happens on the generated .ll.
    println!("issue120_repro: inspect issue120_repro.ll for the dropped Index");
}
