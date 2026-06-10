//! Verbatim from issue #128: `arena_core` sibling crate.
#![no_std]

#[derive(Copy, Clone)]
pub enum Layout {
    Aos,
    Soa,
    AoSoA(u32),
} // payload variant -> align-4, multi-word enum

pub trait CellStore {
    fn bump(&self, n: u32) -> u32;
    fn store(&self, idx: u32, val: u32);
}

// declaration order: store, cap, stride, layout. rustc reorders in memory
// (fat-pointer store first, then u32s, then the enum) -> decl index != LLVM
// field index, and the lowered struct gets [N x i8] padding after the enum.
pub struct ScratchArena<S: CellStore> {
    store: S,
    cap: u32,
    stride: u32,
    layout: Layout,
}

impl<S: CellStore> ScratchArena<S> {
    pub fn new(store: S, cap: u32, stride: u32, layout: Layout) -> Self {
        Self {
            store,
            cap,
            stride,
            layout,
        }
    }
    pub fn alloc(&self) -> Option<u32> {
        let s = self.store.bump(1);
        if s >= self.cap { None } else { Some(s) }
    }
    pub fn write(&self, slot: u32, field: u32, val: u32) {
        let i = match self.layout {
            // reads layout, cap, stride, store off &self
            Layout::Soa => field.wrapping_mul(self.cap).wrapping_add(slot),
            _ => slot.wrapping_mul(self.stride).wrapping_add(field),
        };
        self.store.store(i, val);
    }
}

#[inline(never)] // keep the receiver a pointer-to-struct across the crate boundary
pub fn alloc_and_fill<S: CellStore>(a: &ScratchArena<S>) {
    if let Some(slot) = a.alloc() {
        a.write(slot, 0, slot);
        a.write(slot, 1, slot.wrapping_mul(10));
    }
}
