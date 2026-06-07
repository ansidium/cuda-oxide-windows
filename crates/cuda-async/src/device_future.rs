/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Future type that bridges CUDA stream callbacks with Rust's async executor.
//!
//! [`DeviceFuture`] wraps a [`DeviceOperation`] and drives it through a
//! three-state machine:
//!
//! ```text
//!   Idle ──poll()──> Executing ──callback fires──> Complete
//!                       │                              │
//!                  (enqueue work                  (return result)
//!                   + host callback)
//! ```
//!
//! On the first poll the operation is executed on its assigned stream and a
//! host callback is registered via `cuLaunchHostFunc`. When the GPU reaches
//! the callback, it wakes the future through an [`AtomicWaker`], avoiding
//! busy-waits. If the future is dropped after submission, resources returned
//! by the operation are kept alive until the stream callback fires.
//!
//! [`DeviceOperation`]: crate::device_operation::DeviceOperation
//! [`AtomicWaker`]: futures::task::AtomicWaker

use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::DeviceError;
use futures::task::AtomicWaker;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

/// Lifecycle state of a [`DeviceFuture`].
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
pub enum DeviceFutureState {
    /// The future was constructed in a failed state (e.g. scheduling error).
    Failed,
    /// Initial state: the operation has not yet been submitted to the GPU.
    #[default]
    Idle,
    /// The operation has been submitted; waiting for the stream callback.
    Executing,
    /// The result has been produced. Polling again will panic.
    Complete,
}

/// Shared state between a [`DeviceFuture`] and its `cuLaunchHostFunc` callback.
///
/// The callback sets `complete` and wakes the stored waker, allowing the
/// executor to re-poll the future without busy-waiting.
#[derive(Debug, Default)]
pub struct StreamCallbackState {
    /// Waker registered by the executor during [`Future::poll`].
    pub(crate) waker: AtomicWaker,
    /// Set to `true` by the host callback when the GPU reaches the callback
    /// point in the stream.
    pub(crate) complete: AtomicBool,
}

impl StreamCallbackState {
    /// Creates a new callback state with no registered waker and
    /// `complete = false`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the stream callback as complete and wakes the associated future.
    ///
    /// Called from the `cuLaunchHostFunc` host-side callback.
    pub fn signal(&self) {
        self.complete.store(true, Ordering::Relaxed);
        self.waker.wake();
    }
}

/// A [`Future`] that executes a [`DeviceOperation`] on a CUDA stream and
/// resolves when the GPU signals completion via a host callback.
///
/// Constructed by [`SchedulingPolicy::schedule`] or by the [`IntoFuture`] impl
/// on any `DeviceOperation`.
///
/// [`SchedulingPolicy::schedule`]: crate::scheduling_policies::SchedulingPolicy::schedule
/// [`IntoFuture`]: std::future::IntoFuture
#[derive(Debug)]
pub struct DeviceFuture<T: Send, DO: DeviceOperation<Output = T>> {
    /// The operation to execute. Consumed on first poll.
    pub(crate) device_operation: Option<DO>,
    /// Stream and context for execution. Set by the scheduling policy.
    pub(crate) execution_context: Option<ExecutionContext>,
    /// Holds the result between execution and final poll resolution. The
    /// callback closure owns another Arc so cancellation after submission does
    /// not drop resources still needed by in-flight GPU work.
    pub(crate) result: Option<Arc<Mutex<Option<T>>>>,
    /// Holds an error when the future is in the `Failed` state.
    pub(crate) error: Option<DeviceError>,
    /// Current lifecycle state.
    pub(crate) state: DeviceFutureState,
    /// Shared state with the `cuLaunchHostFunc` callback.
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
}

impl<T: Send, DO: DeviceOperation<Output = T>> DeviceFuture<T, DO> {
    /// Creates an idle future with no operation or context attached.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a future that will immediately yield `Err(error)` on first poll.
    pub fn failed(error: DeviceError) -> Self {
        Self {
            execution_context: None,
            device_operation: None,
            state: DeviceFutureState::Failed,
            callback_state: None,
            result: None,
            error: Some(error),
        }
    }

    /// Registers a `cuLaunchHostFunc` callback that will signal `waker_state`
    /// when the GPU reaches this point in the stream.
    ///
    /// # Safety
    ///
    /// The execution context must hold a valid, non-destroyed CUDA stream.
    unsafe fn register_callback(
        &self,
        waker_state: Arc<StreamCallbackState>,
        result_state: Arc<Mutex<Option<T>>>,
    ) -> Result<(), DeviceError> {
        let ctx = self.execution_context.as_ref().ok_or_else(|| {
            DeviceError::Internal("Cannot execute future without an execution context.".to_string())
        })?;
        ctx.get_cuda_stream().launch_host_function(move || {
            drop(result_state);
            waker_state.signal();
        })?;
        Ok(())
    }

    /// Takes the stored operation, executes it on the bound stream, and stashes
    /// the result. Called exactly once during the `Idle -> Executing` transition.
    fn execute(&mut self) -> Result<(), DeviceError> {
        let ctx = self.execution_context.as_ref().ok_or_else(|| {
            DeviceError::Internal("Cannot execute future without an execution context.".to_string())
        })?;
        let operation = self
            .device_operation
            .take()
            .ok_or_else(|| DeviceError::Internal("No operation has been set.".to_string()))?;
        let out = unsafe { operation.execute(ctx) }?;
        self.result = Some(Arc::new(Mutex::new(Some(out))));
        Ok(())
    }

    /// Best-effort wait before dropping a completed operation's result after
    /// callback registration failed. Without a callback, this is the only
    /// point where the future can keep returned resources alive until the
    /// stream reaches the already-submitted work.
    fn synchronize_after_callback_registration_failure(&self) {
        if let Some(ctx) = &self.execution_context {
            let _ = ctx.get_cuda_stream().synchronize();
        }
    }

    fn take_completed_result(&mut self) -> T {
        let result = self
            .result
            .take()
            .expect("Executing state must carry a result.");
        result
            .lock()
            .expect("DeviceFuture result mutex poisoned")
            .take()
            .expect("Expected result.")
    }
}

impl<T: Send, DO: DeviceOperation<Output = T>> Default for DeviceFuture<T, DO> {
    fn default() -> Self {
        Self {
            device_operation: Default::default(),
            execution_context: Default::default(),
            result: Default::default(),
            error: Default::default(),
            state: Default::default(),
            callback_state: Default::default(),
        }
    }
}

/// `DeviceFuture` does not contain self-referential pointers, so it is safe
/// to move.
impl<T: Send, DO: DeviceOperation<Output = T>> Unpin for DeviceFuture<T, DO> {}

/// State-machine implementation of [`Future`] for CUDA device work.
///
/// | State       | Action on poll                                          |
/// |-------------|---------------------------------------------------------|
/// | `Failed`    | Immediately returns `Err`.                              |
/// | `Idle`      | Executes the operation, registers callback, -> Pending. |
/// | `Executing` | Checks callback flag; returns result if done.           |
/// | `Complete`  | Panics -- must not poll after completion.               |
impl<T: Send, DO: DeviceOperation<Output = T>> Future for DeviceFuture<T, DO> {
    type Output = Result<T, DeviceError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state == DeviceFutureState::Failed {
            self.state = DeviceFutureState::Complete;
            let error = self
                .error
                .take()
                .expect("Failed state must carry an error.");
            return Poll::Ready(Err(error));
        }

        if self.callback_state.is_none() {
            self.callback_state = Some(Arc::new(StreamCallbackState::new()));
        }
        let waker_state = self
            .callback_state
            .as_ref()
            .map(Arc::clone)
            .expect("Impossible.");

        match self.state {
            DeviceFutureState::Idle => {
                waker_state.waker.register(cx.waker());
                if let Err(e) = self.execute() {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                let result_state = self
                    .result
                    .as_ref()
                    .map(Arc::clone)
                    .expect("Executing state must carry a result.");
                if let Err(e) =
                    unsafe { self.register_callback(Arc::clone(&waker_state), result_state) }
                {
                    self.synchronize_after_callback_registration_failure();
                    self.result.take();
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                self.state = DeviceFutureState::Executing;
                Poll::Pending
            }
            DeviceFutureState::Executing => {
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Ok(self.take_completed_result()));
                }
                waker_state.waker.register(cx.waker());
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    Poll::Ready(Ok(self.take_completed_result()))
                } else {
                    Poll::Pending
                }
            }
            DeviceFutureState::Complete => panic!("Poll called after completion."),
            DeviceFutureState::Failed => unreachable!(),
        }
    }
}
