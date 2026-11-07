//! Memory budget accounting.
//!
//! The governor checks the requested total *before* reserving anything. A
//! permit owns its reservation until it is dropped; forgetting a permit keeps
//! the bytes reserved, which is the conservative failure mode.

use std::sync::{Arc, Mutex};

use zkie_core::compute::{
    MemoryGovernor, MemoryPermit, MemoryRequest, MemorySnapshot, PermitId,
};
use zkie_core::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudgetConfig {
    pub budget_bytes: u64,
    pub system_reserve_bytes: u64,
}

impl MemoryBudgetConfig {
    /// Effective budget is the smaller of the configured ceiling and the
    /// currently available bytes after the system reserve. `None` means the
    /// available amount is unknown.
    pub fn effective_budget(&self, available_bytes: Option<u64>) -> u64 {
        match available_bytes {
            Some(available) => self
                .budget_bytes
                .min(available.saturating_sub(self.system_reserve_bytes)),
            None => self.budget_bytes,
        }
    }
}

#[derive(Debug, Default)]
struct Accounting {
    reserved_bytes: u64,
    peak_bytes: u64,
    resident_bytes: u64,
    mapped_window_bytes: u64,
    pinned_bytes: u64,
    metadata_bytes: u64,
    scratch_disk_bytes: u64,
    next_id: u64,
}

#[derive(Clone, Debug)]
pub struct BudgetedMemoryGovernor {
    budget_bytes: u64,
    state: Arc<Mutex<Accounting>>,
}

impl BudgetedMemoryGovernor {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            state: Arc::new(Mutex::new(Accounting::default())),
        }
    }

    pub fn from_config(config: &MemoryBudgetConfig, available_bytes: Option<u64>) -> Self {
        Self::new(config.effective_budget(available_bytes))
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

impl MemoryGovernor for BudgetedMemoryGovernor {
    fn acquire(&self, request: &MemoryRequest) -> Result<MemoryPermit> {
        let total = request.total_bytes()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::SynchronizationFailed)?;
        let remaining = self
            .budget_bytes
            .checked_sub(state.reserved_bytes)
            .ok_or(Error::ResourceLimit)?;
        if total > remaining {
            return Err(Error::ResourceLimit);
        }
        state.reserved_bytes = state
            .reserved_bytes
            .checked_add(total)
            .ok_or(Error::ResourceLimit)?;
        state.peak_bytes = state.peak_bytes.max(state.reserved_bytes);
        state.resident_bytes = state.resident_bytes.saturating_add(request.resident_bytes);
        state.mapped_window_bytes = state
            .mapped_window_bytes
            .saturating_add(request.mapped_window_bytes);
        state.pinned_bytes = state.pinned_bytes.saturating_add(request.pinned_bytes);
        state.metadata_bytes = state.metadata_bytes.saturating_add(request.metadata_bytes);
        state.scratch_disk_bytes = state
            .scratch_disk_bytes
            .saturating_add(request.scratch_disk_bytes);
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or(Error::ResourceLimit)?;
        let id = PermitId(state.next_id);
        let request = *request;
        let state = Arc::clone(&self.state);
        Ok(MemoryPermit::new(id, request, move || {
            if let Ok(mut state) = state.lock() {
                state.reserved_bytes = state.reserved_bytes.saturating_sub(total);
                state.resident_bytes = state
                    .resident_bytes
                    .saturating_sub(request.resident_bytes);
                state.mapped_window_bytes = state
                    .mapped_window_bytes
                    .saturating_sub(request.mapped_window_bytes);
                state.pinned_bytes = state.pinned_bytes.saturating_sub(request.pinned_bytes);
                state.metadata_bytes = state
                    .metadata_bytes
                    .saturating_sub(request.metadata_bytes);
                state.scratch_disk_bytes = state
                    .scratch_disk_bytes
                    .saturating_sub(request.scratch_disk_bytes);
            }
        }))
    }

    fn snapshot(&self) -> MemorySnapshot {
        match self.state.lock() {
            Ok(state) => MemorySnapshot {
                budget_bytes: self.budget_bytes,
                reserved_bytes: state.reserved_bytes,
                peak_bytes: state.peak_bytes,
                resident_bytes: state.resident_bytes,
                mapped_window_bytes: state.mapped_window_bytes,
                pinned_bytes: state.pinned_bytes,
                metadata_bytes: state.metadata_bytes,
                scratch_disk_bytes: state.scratch_disk_bytes,
            },
            Err(_) => MemorySnapshot {
                budget_bytes: self.budget_bytes,
                reserved_bytes: self.budget_bytes,
                peak_bytes: self.budget_bytes,
                resident_bytes: self.budget_bytes,
                mapped_window_bytes: 0,
                pinned_bytes: 0,
                metadata_bytes: 0,
                scratch_disk_bytes: 0,
            },
        }
    }
}
