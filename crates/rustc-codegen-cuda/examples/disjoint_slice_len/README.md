# disjoint_slice_len

Regression test for [issue #343](https://github.com/NVlabs/cuda-oxide/issues/343):
calling `DisjointSlice::len()` inside a kernel must compile and return the launch-time length.

## The bug

`DisjointSlice::len` is intercepted by the mir-importer as an intrinsic (`emit_len`).
Because `len(&self)` receives the slice behind a reference,
the translated operand is a thin `mir.ptr<mir.disjoint_slice<T>>`,
not the fat `(ptr, len)` value.
`emit_len` fed that pointer straight into `mir.extract_field`,
which only accepts the fat value, so device codegen died in dialect verification:

```text
MirExtractFieldOp operand must be tuple, slice, struct, array, or scalar (newtype)
```

The fix validates the `&DisjointSlice<T>` receiver shape and loads the fat
value through that one pointer layer before extracting its length.

## Reproducing the original failure

The interceptor only sees the call when rustc's MIR inliner leaves it intact.
At default release settings rustc inlines `len()` into the kernel and the
pattern disappears. This example therefore declares an extra rustc flag in
its stable-compatible package metadata:

```toml
[package.metadata.cuda-oxide]
extra-rustflags = ["-Zinline-mir=no"]
```

Before the fix, every build of this example failed with the verification
error above. After the fix, it compiles and runs identically to an inlined
build.

`cargo oxide` reads this metadata and applies the flag only when building this
example. Direct Cargo commands such as `cargo fmt`, `cargo metadata`, and
`cargo clippy` continue to work on the stable toolchain.

## What the kernel checks

Every in-bounds thread writes `len()` to its slot. The buffer has 257 elements,
deliberately avoiding a typical block-size multiple so launch geometry cannot
masquerade as the slice length. The host asserts all 257 results.

## Run

```bash
cargo oxide run disjoint_slice_len
```

Expected output:

```text
SUCCESS: DisjointSlice::len returns the launch-time length
```
