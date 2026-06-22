/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer};

#[test]
fn device_buffer_from_host_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [1_u32, 2, 3, 4, 5];
    let dev_buf =
        DeviceBuffer::from_host(&stream, &data).expect("failed to allocate DeviceBuffer from host");

    assert_eq!(dev_buf.len(), 5);
    assert_eq!(dev_buf.num_bytes(), 20);
    assert!(!dev_buf.is_empty());

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, data);
}

#[test]
fn device_buffer_zeroed_initializes_with_zeros() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<f32>::zeroed(&stream, 4).expect("failed to allocate zeroed DeviceBuffer");

    assert_eq!(dev_buf.len(), 4);
    assert_eq!(dev_buf.num_bytes(), 16);

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn device_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<u8>::zeroed(&stream, 0).expect("failed to allocate empty device buffer");
    assert_eq!(dev_buf.len(), 0);
    assert_eq!(dev_buf.num_bytes(), 0);
    assert!(dev_buf.is_empty());

    let dev_buf_host = DeviceBuffer::<u8>::from_host(&stream, &[])
        .expect("failed to allocate empty device buffer from empty slice");
    assert_eq!(dev_buf_host.len(), 0);
    assert_eq!(dev_buf_host.num_bytes(), 0);
    assert!(dev_buf_host.is_empty());
}

#[test]
fn device_buffer_rejects_allocation_size_overflow() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let overflowing_len = usize::MAX / std::mem::size_of::<u64>() + 1;

    assert!(DeviceBuffer::<u64>::zeroed(&stream, overflowing_len).is_err());
    // SAFETY: the constructor returns an error before allocation, and this
    // test never reads from the uninitialized buffer.
    assert!(unsafe { DeviceBuffer::<u64>::uninitialized_async(&stream, overflowing_len) }.is_err());
}

#[test]
fn device_buffer_async_compat_methods_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [7_u32, 11, 13, 17];
    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate uninitialized device buffer");
    // SAFETY: `data` remains alive and unmodified until the later
    // `to_host_vec` call synchronizes the stream.
    unsafe { dev.copy_from_host_async_unchecked(&stream, &data) }
        .expect("failed to copy host data into device buffer");

    let mut clone = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate clone device buffer");
    clone
        .copy_from_device_async(&dev, &stream)
        .expect("failed to copy device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy clone back to host"),
        data
    );

    clone
        .zero_async(&stream)
        .expect("failed to zero device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy zeroed buffer back to host"),
        [0, 0, 0, 0]
    );

    clone
        .drop_async(&stream)
        .expect("failed to async free clone");
    dev.drop_async(&stream)
        .expect("failed to async free source");

    let empty = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 0) }
        .expect("failed to allocate empty uninitialized device buffer");
    empty
        .drop_async(&stream)
        .expect("failed to async free empty buffer");
    stream.synchronize().expect("stream sync failed");
}

#[test]
fn from_host_with_pinned_source_allows_source_drop_after_return() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let expected = vec![21_u32, 34, 55, 89];
    let input =
        PinnedHostBuffer::from_slice(&ctx, &expected).expect("failed to allocate pinned input");
    let dev = DeviceBuffer::from_host(&stream, input.as_slice())
        .expect("failed to copy pinned input to device");
    drop(input);

    assert_eq!(
        dev.to_host_vec(&stream)
            .expect("failed to copy device buffer back to host"),
        expected
    );
}

#[test]
fn copy_from_host_with_pinned_source_allows_source_reuse_after_return() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let expected = vec![3_u32, 5, 8, 13];
    let mut input =
        PinnedHostBuffer::from_slice(&ctx, &expected).expect("failed to allocate pinned input");
    let mut dev =
        DeviceBuffer::<u32>::zeroed(&stream, input.len()).expect("failed to allocate device");

    dev.copy_from_host(&stream, input.as_slice())
        .expect("failed to copy pinned input to device");
    input.as_mut_slice().fill(0);

    assert_eq!(
        dev.to_host_vec(&stream)
            .expect("failed to copy device buffer back to host"),
        expected
    );
}

// Dangerous-path regression: a stream-ordered (`cuMemAllocAsync`) buffer that
// is dropped *implicitly* while a copy is still in flight, with NO explicit
// synchronization. The buffer co-owns its stream, so `Drop` frees it with
// `cuMemFreeAsync` on that same stream (stream-ordered, race-free) instead of
// the synchronous `cuMemFree`. Run under `compute-sanitizer --tool memcheck`
// to catch a regression: pairing async-pool allocation with the synchronous
// free reports "free-before-alloc" here.
#[test]
fn uninitialized_async_implicit_drop_is_stream_ordered() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let n = 1 << 20; // 4 MiB of u32
    let src = DeviceBuffer::<u32>::zeroed(&stream, n).expect("failed to allocate source buffer");
    for _ in 0..64 {
        let mut dst = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, n) }
            .expect("failed to allocate uninitialized device buffer");
        // Enqueue a large async device-to-device copy, then let `dst` drop
        // immediately with no `stream.synchronize()` in between.
        dst.copy_from_device_async(&src, &stream)
            .expect("failed to enqueue device-to-device copy");
        drop(dst);
    }
    stream.synchronize().expect("stream sync failed");
}

#[test]
fn uninitialized_async_cast_elem_implicit_drop_is_stream_ordered() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let n = 1 << 20; // 4 MiB of u32
    let src = DeviceBuffer::<u32>::zeroed(&stream, n).expect("failed to allocate source buffer");
    for _ in 0..64 {
        let mut dst = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, n) }
            .expect("failed to allocate uninitialized device buffer");
        dst.copy_from_device_async(&src, &stream)
            .expect("failed to enqueue device-to-device copy");
        let dst = dst.cast_elem::<std::num::Wrapping<u32>>();
        drop(dst);
    }
    stream.synchronize().expect("stream sync failed");
}
