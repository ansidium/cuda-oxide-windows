//! Independent repro for issue #128: kernel silently reads the wrong struct
//! field (reordered/padded repr(Rust) struct), cross-crate variant.
//!
//! `ScratchArena<DeviceStore>` is declared (store, cap, stride, layout) in
//! the sibling `arena_core` crate; rustc reorders it in memory and the
//! lowered LLVM struct gains interior `[N x i8]` padding after the enum.
//! Every field access in `alloc_and_fill` goes through `&self`, exercising
//! the decl-index -> LLVM-index mapping in mir-lower's aggregate lowering.
//!
//! Expected per the issue: either an LLVM-verification / lowering error
//! (e.g. "IntToInt: source is not an integer") or, on the silent path,
//! wrong field indices in the emitted .ll.

use arena_core::{CellStore, Layout, ScratchArena, alloc_and_fill};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::kernel;

/// CellStore backed by a device buffer + atomic cursor
/// (bump = atomic fetch_add, store = raw write), as described in the issue.
struct DeviceStore<'a> {
    cursor: &'a DeviceAtomicU32,
    data: &'a [u32],
}

impl<'a> CellStore for DeviceStore<'a> {
    fn bump(&self, n: u32) -> u32 {
        self.cursor.fetch_add(n, AtomicOrdering::Relaxed)
    }
    fn store(&self, idx: u32, val: u32) {
        unsafe {
            let p = self.data.as_ptr() as *mut u32;
            p.add(idx as usize).write(val);
        }
    }
}

#[kernel]
pub fn fill(cursor: &[u32], data: &[u32], params: &[u32]) {
    let cur = unsafe { &*(cursor.as_ptr() as *const DeviceAtomicU32) };
    let store = DeviceStore { cursor: cur, data }; // impl CellStore: bump = atomic fetch_add, store = raw write
    let arena = ScratchArena::new(store, params[0] /*cap*/, params[1] /*stride*/, Layout::Soa);
    alloc_and_fill(&arena); // every field access here goes through the mis-mapped index
}

fn main() {
    // Compile-only repro; no GPU execution in this environment.
    println!("issue128_repro2: device compilation check only");
}
