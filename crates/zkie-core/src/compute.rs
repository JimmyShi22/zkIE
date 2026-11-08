//! Compute and resource ports.
//!
//! The ports are intentionally small and object-safe. Concrete backends live in
//! the engine layer; operators and proof code only see these traits.

use std::fmt;

use crate::common::codec::DecodeLimits;
use crate::common::field::ProtocolField;
use crate::error::{Error, Result};
use crate::progress::ProgressSink;
use crate::storage::TensorStorageService;
use crate::types::{
    BufferHandle, BufferSpec, ElementRange, Fence, TensorSpec, TensorView, WorkspaceRequirement,
};

/// Bytes requested from the memory governor before any allocation or mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryRequest {
    pub resident_bytes: u64,
    pub mapped_window_bytes: u64,
    pub pinned_bytes: u64,
    pub metadata_bytes: u64,
    pub scratch_disk_bytes: u64,
}

impl MemoryRequest {
    pub fn total_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for bytes in [
            self.resident_bytes,
            self.mapped_window_bytes,
            self.pinned_bytes,
            self.metadata_bytes,
            self.scratch_disk_bytes,
        ] {
            total = total.checked_add(bytes).ok_or(Error::ResourceLimit)?;
        }
        Ok(total)
    }

    pub fn validate(&self) -> Result<()> {
        self.total_bytes().map(|_| ())
    }

    pub fn from_workspace(workspace: &WorkspaceRequirement) -> Self {
        Self {
            resident_bytes: workspace.selected_resident_bytes,
            mapped_window_bytes: workspace.mapped_window_bytes,
            scratch_disk_bytes: workspace.scratch_disk_bytes,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub budget_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_bytes: u64,
    pub resident_bytes: u64,
    pub mapped_window_bytes: u64,
    pub pinned_bytes: u64,
    pub metadata_bytes: u64,
    pub scratch_disk_bytes: u64,
}

impl MemorySnapshot {
    pub fn remaining_bytes(&self) -> u64 {
        self.budget_bytes.saturating_sub(self.reserved_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PermitId(pub u64);

/// Move-only RAII permit. Dropping it invokes the release hook; forgetting it
/// conservatively keeps the budget reserved.
pub struct MemoryPermit {
    id: PermitId,
    request: MemoryRequest,
    release: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl MemoryPermit {
    pub fn new(
        id: PermitId,
        request: MemoryRequest,
        release: impl FnOnce() + Send + Sync + 'static,
    ) -> Self {
        Self {
            id,
            request,
            release: Some(Box::new(release)),
        }
    }

    pub const fn id(&self) -> PermitId {
        self.id
    }

    pub const fn request(&self) -> &MemoryRequest {
        &self.request
    }
}

impl Drop for MemoryPermit {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

impl fmt::Debug for MemoryPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPermit")
            .field("id", &self.id)
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

pub trait MemoryGovernor: Send + Sync {
    fn acquire(&self, request: &MemoryRequest) -> Result<MemoryPermit>;
    fn snapshot(&self) -> MemorySnapshot;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub resident_bytes: u64,
    pub mapped_bytes: u64,
    pub scratch_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeBackendId(pub u32);

impl RuntimeBackendId {
    pub const CPU: Self = Self(1);
    pub const GPU: Self = Self(2);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeCapabilities {
    pub exact_integer: bool,
    pub base_field: bool,
    pub extension_field: bool,
    pub gpu: bool,
}

impl ComputeCapabilities {
    /// Conservative capability report: only the base field is available until
    /// the exact integer kernels and the extension field are implemented.
    pub const fn cpu() -> Self {
        Self {
            exact_integer: false,
            base_field: true,
            extension_field: false,
            gpu: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub u32);

#[derive(Clone, Debug)]
pub struct KernelJob<F: ProtocolField> {
    pub kernel: KernelId,
    pub params: Vec<F>,
}

impl<F: ProtocolField> KernelJob<F> {
    pub fn new(kernel: KernelId, params: Vec<F>) -> Self {
        Self { kernel, params }
    }
}

pub trait ComputeBackend<F: ProtocolField>: Send {
    fn identity(&self) -> RuntimeBackendId;
    fn capabilities(&self) -> ComputeCapabilities;

    fn allocate(&mut self, spec: &BufferSpec, permit: &MemoryPermit) -> Result<BufferHandle>;

    fn upload(&mut self, dst: BufferHandle, canonical_bytes: &[u8]) -> Result<Fence>;

    fn download_range(
        &mut self,
        src: BufferHandle,
        range: ElementRange,
        wait_for: &[Fence],
        dst: &mut [u8],
    ) -> Result<()>;

    fn submit(
        &mut self,
        job: &KernelJob<F>,
        inputs: &[BufferHandle],
        outputs: &[BufferHandle],
        wait_for: &[Fence],
    ) -> Result<Fence>;

    fn wait(&mut self, fence: Fence) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
    fn release(&mut self, buffer: BufferHandle) -> Result<()>;
    fn memory_usage(&self) -> MemoryUsage;
}

/// The execution boundary used by operators and engine kernels.
pub trait ExecutionPort<F: ProtocolField> {
    fn storage(&self) -> &dyn TensorStorageService;
    fn memory(&self) -> &dyn MemoryGovernor;
    fn progress(&self) -> &dyn ProgressSink;

    fn execute(
        &mut self,
        job: &KernelJob<F>,
        inputs: &[TensorView],
        outputs: &[TensorSpec],
        workspace: &WorkspaceRequirement,
    ) -> Result<Vec<TensorView>>;
}

/// Public verification limits. These are runtime policy, not protocol math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierPolicy {
    pub max_proof_bytes: u64,
    pub max_decode_elements: u64,
    pub require_exact_integer: bool,
}

impl Default for VerifierPolicy {
    fn default() -> Self {
        Self {
            max_proof_bytes: 1 << 30,
            max_decode_elements: 1 << 20,
            require_exact_integer: true,
        }
    }
}

impl VerifierPolicy {
    pub fn decode_limits(&self) -> DecodeLimits {
        DecodeLimits {
            max_bytes: usize::try_from(self.max_proof_bytes).unwrap_or(usize::MAX),
            max_elements: usize::try_from(self.max_decode_elements).unwrap_or(usize::MAX),
        }
    }
}
