//! Repro for issue #131: `match xs[i]` over an array of enums fails to
//! import with "Downcast on non-ADT type: Array".
//!
//! The place for the match payload binding is `xs[i]` with the projection
//! chain [Index(i), Downcast(variant), Field(0)]. The value-producing place
//! walker (`translate_place_iterative` in mir-importer) never narrows the
//! running Rust type on Index/ConstantIndex, so Downcast/Field see the
//! outer Array type and bail.
//!
//! Note: a fully-constant `match xs[0]` over a constant array is folded
//! away by rustc's MIR optimizations before it reaches the importer, so the
//! constant-index kernel below derives the payload from a kernel parameter
//! to keep the projection chain alive.

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
pub enum E {
    A(u32),
    B(u32),
    C,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Runtime index: projection chain [Index(local), Downcast, Field].
    #[kernel]
    pub fn match_runtime_index(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 4] = [E::A(7), E::B(8), E::C, E::A(100)];
            *out_elem = match xs[index as usize] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
        }
    }

    /// Literal index with a runtime-unknown discriminant: exercises the
    /// constant-index flavor of the same stale-type walk.
    #[kernel]
    pub fn match_const_index(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 2] = if val > 5 {
                [E::A(val), E::C]
            } else {
                [E::B(val), E::C]
            };
            *out_elem = match xs[0] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
        }
    }
}

fn main() {
    // Compile-only repro: no GPU execution required. The bug fires during
    // device codegen (MIR import), not at runtime.
    println!("issue131_repro host binary built");
}
