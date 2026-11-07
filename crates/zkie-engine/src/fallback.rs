//! Recoverable CPU compute transactions.
//!
//! A retry reruns the same authenticated input. It must not request new
//! protocol randomness, advance a transcript, or change an already published
//! message.

use zkie_core::error::{Error, Result};

use crate::context::FallbackPolicy;

pub fn is_recoverable_compute(error: &Error) -> bool {
    matches!(
        error,
        Error::SynchronizationFailed | Error::DeviceLost | Error::DeviceOutOfMemory
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeTransaction {
    retries: u32,
}

impl ComputeTransaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retries(&self) -> u32 {
        self.retries
    }

    pub fn run<T, F>(&mut self, policy: FallbackPolicy, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        match operation() {
            Ok(value) => Ok(value),
            Err(error)
                if policy == FallbackPolicy::Recompute
                    && self.retries == 0
                    && is_recoverable_compute(&error) =>
            {
                self.retries += 1;
                operation()
            }
            Err(error) => Err(error),
        }
    }
}
