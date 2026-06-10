//! Independent repro for issue #132: kernel hits an illegal memory access
//! (CUDA 700) when it uses `Option<&T>::unwrap_or(&literal)`.
//!
//! Kernel is the reporter's code verbatim. With both a `Some` and a `None`
//! of `Option<&u32>` live in one function, `-O` const-folds the `None`
//! arm's `unwrap_or(&77)` into a reference-to-scalar constant. Per the
//! report, the importer drops that constant's provenance and emits
//! `inttoptr i64 0` + a load through it, so `out[1]` faults (or reads
//! garbage) instead of being 77.
//!
//! Compile-only verification: launch expectation is out == [5, 77].

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn opt_ref_unwrap_or(out: &[u32]) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let r: u32 = 5;
        let a: Option<&u32> = Some(&r);
        let b: Option<&u32> = None; // keeping BOTH a Some and a None live is what
        // makes -O const-fold the None arm's unwrap_or
        // into the `&77` reference-to-scalar constant
        let v0: u32 = *a.unwrap_or(&77); // 5
        let v1: u32 = *b.unwrap_or(&77); // should be 77 — pre-fix this lowers to load i32, ptr null

        unsafe {
            // out is &[u32] aliasing a writable buffer;
            // the raw write is incidental to the bug
            let p = out.as_ptr() as *mut u32;
            *p.add(0) = v0;
            *p.add(1) = v1;
        }
    }
}

fn main() {
    // Compile-only repro: no GPU needed here. The miscompile is visible in
    // the generated .ll (null-pointer load where 77 should be read).
    println!("issue132_repro2 host binary built");
}
