/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Repro for issue #146: kernel does not behave correctly with some uses of enum.
//!
//! `helper` returns `Ordering::Less` (discriminant -1, variant index 0).
//! The `match` on the returned value takes the `Equal` arm instead, so the
//! output cell stays 0 instead of becoming -1.
//!
//! Expected (correct compiler): out == [-1]
//! Observed (buggy):            out == [0]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, gpu_printf, kernel, thread};
    use std::cmp::Ordering;

    #[kernel]
    pub fn misbehave(mut dst: DisjointSlice<i32>) {
        let idx = thread::index_1d();

        if let Some(cell) = dst.get_mut(idx) {
            let lhs = [0, 0x169000, 0, 0];
            let rhs = [0, 0xbd1000, 0, 0];

            let comparison = helper(lhs, rhs);

            match comparison {
                Ordering::Less => {
                    *cell = -1;
                }
                Ordering::Equal => {
                    *cell = 0;
                }
                Ordering::Greater => {
                    *cell = 1;
                }
            }
        }
    }

    fn helper(lhs: [i32; 4], rhs: [i32; 4]) -> Ordering {
        for pos in 0..4 {
            gpu_printf!("{} <=> {}\n", lhs[pos], rhs[pos]);
            if lhs[pos] < rhs[pos] {
                gpu_printf!("less\n");
                return Ordering::Less;
            }
            if lhs[pos] > rhs[pos] {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
}

fn main() {
    let (stream, module) = simple_stream_and_module();

    let mut c_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();

    module
        .misbehave(&stream, LaunchConfig::for_num_elems(1), &mut c_dev)
        .unwrap();
    stream.context().synchronize().unwrap();
    let c_host = c_dev.to_host_vec(&stream).unwrap();

    println!("{c_host:?}");
    if c_host == [-1] {
        println!("SUCCESS");
    } else {
        println!("FAIL: expected [-1], got {c_host:?}");
    }
}

fn simple_stream_and_module() -> (Arc<CudaStream>, kernels::LoadedModule) {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");
    (stream, module)
}
