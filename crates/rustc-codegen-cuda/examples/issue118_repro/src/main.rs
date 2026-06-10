//! Repro for issue #118: fieldless `#[repr(u32)]` enum lowered with a
//! discriminant width inferred from the variant count instead of the
//! explicit repr. Compile-only check: inspect issue118_repro.ll for the
//! GEP element type used by `*const Tag` pointer arithmetic (`.add(i)`).
//! Correct Rust semantics: size_of::<Tag>() == 4, so the stride must be
//! 4 bytes (i32/u32 element), not 1 byte (i8).

use cuda_device::{DisjointSlice, kernel, thread};

/// Fieldless `#[repr(u32)]` enum used as a tag-like device-buffer element.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Tag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

const N: usize = 4;

/// Control: read via `*const u32` with `add(i)`. Stride should be 4.
#[kernel]
pub fn read_via_u32(input: &[u32], mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= N {
        return;
    }
    let base: *const u32 = input.as_ptr();
    let v = unsafe { *base.add(i) };
    if let Some(slot) = out.get_mut(idx) {
        *slot = v;
    }
}

/// Test: read via `*const Tag` with `add(i)`, then cast the discriminant
/// back to u32. If stride is correctly 4 the output matches the u32
/// control. If stride is buggy the output reads the wrong bytes.
#[kernel]
pub fn read_via_enum(input: &[u32], mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= N {
        return;
    }
    let base: *const Tag = input.as_ptr() as *const Tag;
    let tag = unsafe { *base.add(i) };
    if let Some(slot) = out.get_mut(idx) {
        *slot = tag as u32;
    }
}

fn main() {
    // Compile-only repro; no GPU execution in this environment.
    println!("issue118_repro: device compilation check only");
}
