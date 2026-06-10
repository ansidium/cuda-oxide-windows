//! Independent repro (repro2) for issue #123.
//!
//! Rust semantics: `NaN != NaN` is `true` (PartialEq::ne on floats is
//! unordered). Correct LLVM lowering of float `!=` is `fcmp une`.
//! The issue reports cuda-oxide emits `fcmp one` (ordered), which is
//! false when either operand is NaN, so `v != v` can never detect NaN.
//!
//! Kernel below is the reporter's original code, verbatim.

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn is_nan(x: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(ce) = c.get_mut(idx) {
        let v = x[i];
        *ce = if v != v { 1.0 } else { 0.0 };
    }
}

fn main() {
    // Compile-only repro; evidence lives in the generated .ll / .ptx.
    println!("issue123_repro2: inspect issue123_repro2.ll for the fcmp predicate");
}
