/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster Launch Control (CLC) intrinsics for Blackwell+ (SM 100+).
//!
//! CLC replaces software atomic counters for persistent kernel tile scheduling.
//! Instead of a fixed number of persistent CTAs competing on a global counter,
//! CLC uses hardware-managed work-stealing: launch a full grid (one CTA per tile),
//! and running CTAs call `try_cancel` to steal not-yet-launched tiles.
//!
//! # How It Works
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  Hardware Scheduler                                     │
//! │       │                                                 │
//! │       │  Launches CTAs as SMs free up                   │
//! │       ▼                                                 │
//! │  ┌─────────────┐   try_cancel    ┌───────────────────┐  │
//! │  │  Running CTA│───────────────► │ Pending CTA queue │  │
//! │  │  (wants more│◄─────────────── │ (not yet launched)│  │
//! │  │   work)     │  16B response   │                   │  │
//! │  └─────────────┘                 └───────────────────┘  │
//! │       │                                                 │
//! │       │ query_cancel                                    │
//! │       ▼                                                 │
//! │  is_canceled? → if yes: use ctaid as next tile          │
//! │               → if no: exit (no more work)              │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Response Format
//!
//! `try_cancel` writes a 16-byte (b128) response to shared memory. The response
//! is decoded with `query_cancel` variants to extract:
//! - Whether a CTA was successfully canceled (i.e., work is available)
//! - The grid coordinates (ctaid.x, ctaid.y, ctaid.z) of the canceled CTA
//!
//! The response is passed as two `u64` values (lo, hi) to the query functions,
//! which pack them into a `.b128` PTX register internally.
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::clc::*;
//! use cuda_device::barrier::{Barrier, mbarrier_init};
//!
//! // Shared memory for the 16-byte response and mbarrier
//! static mut CLC_RESPONSE: [u64; 2] = [0; 2];
//! static mut CLC_BAR: Barrier = Barrier::UNINIT;
//!
//! // Request next tile
//! unsafe {
//!     clc_try_cancel(
//!         &raw mut CLC_RESPONSE as *mut u8,
//!         &raw mut CLC_BAR as *mut u64,
//!     );
//! }
//!
//! // Wait for response via mbarrier...
//!
//! // Decode the response
//! let lo = unsafe { CLC_RESPONSE[0] };
//! let hi = unsafe { CLC_RESPONSE[1] };
//! let canceled = unsafe { clc_query_is_canceled(lo, hi) };
//! if canceled != 0 {
//!     // Work available — get tile coordinates
//!     let tile_x = unsafe { clc_query_get_first_ctaid_x(lo, hi) };
//!     let tile_y = unsafe { clc_query_get_first_ctaid_y(lo, hi) };
//!     // ... process tile (tile_x, tile_y)
//! } else {
//!     // No more work — exit
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **PTX ISA 8.6+**
//! - **SM 100+ (Blackwell)**

use crate::barrier::Barrier;

// =============================================================================
// try_cancel: Request to steal a pending CTA's work
// =============================================================================

/// Asynchronously request to cancel a not-yet-launched CTA and steal its work.
///
/// The hardware writes a 16-byte response to `response` (in shared memory)
/// and signals `mbar` with 16 tx bytes when the response is ready.
///
/// # Parameters
///
/// - `response`: Pointer to 16-byte aligned shared memory for the response
/// - `mbar`: Pointer to an initialized mbarrier (with tx_count covering 16 bytes)
///
/// # Safety
///
/// - `response` must point to at least 16 bytes of shared memory, 16-byte aligned
/// - `mbar` must be a valid, initialized mbarrier in shared memory
/// - Must be called from a cluster-launched kernel on SM 100+
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
///     [response], [mbar];
/// ```
#[inline(never)]
pub unsafe fn clc_try_cancel(response: *mut u8, mbar: *mut Barrier) {
    let _ = (response, mbar);
    unreachable!("clc_try_cancel called outside CUDA kernel context")
}

/// Multicast variant of `try_cancel` — broadcasts response to all CTAs in the cluster.
///
/// Same as [`clc_try_cancel`] but the 16-byte response is delivered to all CTAs
/// in the cluster simultaneously, not just the calling CTA.
///
/// # Safety
///
/// - Same requirements as `clc_try_cancel`
/// - Requires `sm_100a` or `sm_110a` (architecture-accelerated variants)
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes
///     .multicast::cluster::all.b128 [response], [mbar];
/// ```
#[inline(never)]
pub unsafe fn clc_try_cancel_multicast(response: *mut u8, mbar: *mut Barrier) {
    let _ = (response, mbar);
    unreachable!("clc_try_cancel_multicast called outside CUDA kernel context")
}

// =============================================================================
// query_cancel: Decode the 16-byte response
// =============================================================================

/// Check whether a pending CTA was successfully canceled (stolen).
///
/// Returns 1 if a CTA was canceled (work IS available — decode coordinates),
/// 0 if no CTA was canceled (no more work — exit the persistent loop).
///
/// # Parameters
///
/// - `resp_lo`: Low 64 bits of the 16-byte response
/// - `resp_hi`: High 64 bits of the 16-byte response
///
/// # Safety
///
/// - `resp_lo` and `resp_hi` must come from a completed `try_cancel` response
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred, response;
/// ```
#[inline(never)]
pub unsafe fn clc_query_is_canceled(resp_lo: u64, resp_hi: u64) -> u32 {
    let _ = (resp_lo, resp_hi);
    unreachable!("clc_query_is_canceled called outside CUDA kernel context")
}

/// Get the X coordinate of the canceled CTA's grid position.
///
/// Only valid when `clc_query_is_canceled` returned 1 (work available).
///
/// # Safety
///
/// Must be called from a CUDA kernel context. The response pair must
/// originate from a prior `clc_query_cancel` call.
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret, response;
/// ```
#[inline(never)]
pub unsafe fn clc_query_get_first_ctaid_x(resp_lo: u64, resp_hi: u64) -> u32 {
    let _ = (resp_lo, resp_hi);
    unreachable!("clc_query_get_first_ctaid_x called outside CUDA kernel context")
}

/// Get the Y coordinate of the canceled CTA's grid position.
///
/// Only valid when `clc_query_is_canceled` returned 1 (work available).
///
/// # Safety
///
/// Must be called from a CUDA kernel context. The response pair must
/// originate from a prior `clc_query_cancel` call.
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 ret, response;
/// ```
#[inline(never)]
pub unsafe fn clc_query_get_first_ctaid_y(resp_lo: u64, resp_hi: u64) -> u32 {
    let _ = (resp_lo, resp_hi);
    unreachable!("clc_query_get_first_ctaid_y called outside CUDA kernel context")
}

/// Get the Z coordinate of the canceled CTA's grid position.
///
/// Only valid when `clc_query_is_canceled` returned 1 (work available).
///
/// # Safety
///
/// Must be called from a CUDA kernel context. The response pair must
/// originate from a prior `clc_query_cancel` call.
///
/// # PTX
///
/// ```ptx
/// clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 ret, response;
/// ```
#[inline(never)]
pub unsafe fn clc_query_get_first_ctaid_z(resp_lo: u64, resp_hi: u64) -> u32 {
    let _ = (resp_lo, resp_hi);
    unreachable!("clc_query_get_first_ctaid_z called outside CUDA kernel context")
}
