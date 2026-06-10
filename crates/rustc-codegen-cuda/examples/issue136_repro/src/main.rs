/*
 * Repro for issue #136: "Unsupported construct: BinaryOp Cmp not yet implemented"
 *
 * A custom `Ord` impl whose body delegates to integer `Ord::cmp` (the
 * standard pattern for hand-written `cmp`) produces MIR `BinOp::Cmp`
 * (three-way comparison returning core::cmp::Ordering). The mir-importer
 * translator's BinaryOp match has no arm for `BinOp::Cmp` and bails.
 */

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};
    use std::cmp::Ordering;

    #[derive(Eq, PartialEq, PartialOrd, Default)]
    struct Foo<T> {
        pieces: [T; 4],
    }

    impl<T> Ord for Foo<T>
    where
        T: Ord,
    {
        fn cmp(&self, other: &Self) -> Ordering {
            self.pieces[0].cmp(&other.pieces[0])
        }
    }

    #[kernel]
    pub fn asplode(mut dst: DisjointSlice<i32>) {
        let idx = thread::index_1d();

        let x = Foo::<u32>::default().cmp(&Foo::default());

        if let Some(out) = dst.get_mut(idx) {
            *out = x as i32;
        }
    }
}

fn main() {
    // Compile-only repro: device codegen is exercised at build time.
    println!("issue136_repro host main (no kernel launch)");
}
