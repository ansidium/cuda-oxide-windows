//! Repro for issue #128: kernel silently reads the wrong struct field for a
//! reordered/padded repr(Rust) struct.
//!
//! `Arena` declares fields (layout, cap, stride, big). rustc reorders `big`
//! (align 8) to offset 0, and the enum field `Layout` occupies 8 bytes in
//! rustc's layout but only 5 bytes in the lowered `{ i8, i32 }` form, so the
//! padded LLVM struct gains an interior `[3 x i8]` slot:
//!
//!   rustc:  big@0, layout@8, cap@16, stride@20, size 24
//!   LLVM:   { i64, { i8, i32 }, [3 x i8], i32, i32 }
//!            idx0  idx1         idx2      idx3 idx4
//!
//! Correct field indices: big=0, layout=1, cap=3, stride=4. mir-lower's
//! aggregate sites count only real (non-pad) fields, yielding cap=2 (the
//! padding slot) and stride=3 (cap's slot): either an LLVM verification
//! failure on insertvalue or a silent wrong-field read through GEP.
//!
//! Compile-only check: inspect issue128_repro.ll.

use cuda_device::{DisjointSlice, kernel, thread};

#[derive(Copy, Clone)]
pub enum Layout {
    Aos,
    Soa,
    AoSoA(u32), // payload variant -> multi-word, align-4 enum
}

// Declaration order: layout, cap, stride, big.
// rustc memory order: big (align 8) first, then layout, cap, stride.
pub struct Arena {
    layout: Layout,
    cap: u32,
    stride: u32,
    big: u64,
}

/// Keep the receiver a pointer-to-struct so field reads go through
/// mir.field_addr (GEP) instead of being SROA'd away.
#[inline(never)]
fn pick(a: &Arena) -> u32 {
    match a.layout {
        Layout::Soa => a
            .cap
            .wrapping_mul(1000)
            .wrapping_add(a.stride)
            .wrapping_add(a.big as u32),
        Layout::Aos => a.stride,
        Layout::AoSoA(w) => w,
    }
}

#[kernel]
pub fn fill(params: &[u32], mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        let arena = Arena {
            layout: Layout::Soa,
            cap: params[0],
            stride: params[1],
            big: 7,
        };
        // Expected on GPU: params[0]*1000 + params[1] + 7
        *slot = pick(&arena);
    }
}

fn main() {
    // Compile-only repro; no GPU execution in this environment.
    println!("issue128_repro: device compilation check only");
}
