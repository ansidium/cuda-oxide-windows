//! Independent repro for issue #118: MIR importer lowers explicit repr enum
//! discriminants with the wrong width.
//!
//! Kernels are kept verbatim from the issue report. The host `main` is
//! trivial because this repro is verified compile-only (.ll/.ptx inspection).

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Fieldless `#[repr(u32)]` enum used as a tag-like device-buffer element.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Tag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

// SAFETY: trivial repr(u32) POD, safe to copy device-host.
unsafe impl cuda_core::DeviceCopy for Tag {}

const N: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Control: read via `*const u32` with `add(i)`. Stride should be 4.
    /// Writes input[i] for i in 0..N.
    #[kernel]
    pub fn read_via_u32(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Read through a raw u32 pointer with arithmetic.
        let base: *const u32 = input.as_ptr();
        let v = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// Test: read via `*const Tag` with `add(i)`, then cast the discriminant
    /// back to u32. If stride is correctly 4 the output matches the u32
    /// control. If stride is buggy (8) the output skips slots.
    #[kernel]
    pub fn read_via_enum(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Reinterpret the input buffer as `*const Tag`. The bytes are
        // identical (both u32-sized, repr(u32)); only pointer-arithmetic
        // stride is under test.
        let base: *const Tag = input.as_ptr() as *const Tag;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }
}

fn main() {
    // Compile-only repro: no GPU available in this environment.
    // Device-side verification happens by inspecting the generated
    // issue118_repro2.ll / issue118_repro2.ptx artifacts.
    println!("issue118_repro2: compile-only repro for issue #118");
}
