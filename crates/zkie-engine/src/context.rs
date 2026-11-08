//! Execution context.
//!
//! The context borrows the compute, storage, memory, progress and measurement
//! services. It does not own a thread pool and does not hold protocol state;
//! protocol state and randomness live in the proof pipeline.

use std::collections::BTreeMap;

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::ProtocolField;
use zkie_core::compute::{
    ComputeBackend, ComputeCapabilities, ExecutionPort, KernelJob, MemoryGovernor,
    MemoryRequest, VerifierPolicy,
};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressDetail, ProgressSink};
use zkie_core::storage::TensorStorageService;
use zkie_core::types::{
    BufferSpec, ElementRange, TensorId, TensorSpec, TensorStorageRef, TensorView,
    WorkspaceRequirement,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeSelection {
    #[default]
    Cpu,
    RequireGpu,
    Auto,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FallbackPolicy {
    #[default]
    Recompute,
    Fail,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub compute: ComputeSelection,
    pub compute_threads: usize,
    pub memory_budget_bytes: u64,
    pub system_reserve_bytes: u64,
    pub scratch_budget_bytes: u64,
    pub mmap_window_bytes: u64,
    pub io_queue_depth: usize,
    pub progress_detail: ProgressDetail,
    pub fallback: FallbackPolicy,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let compute_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            compute: ComputeSelection::Cpu,
            compute_threads,
            memory_budget_bytes: 1 << 30,
            system_reserve_bytes: 0,
            scratch_budget_bytes: 0,
            mmap_window_bytes: 0,
            io_queue_depth: 1,
            progress_detail: ProgressDetail::Stage,
            fallback: FallbackPolicy::Recompute,
        }
    }
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.compute_threads == 0 || self.io_queue_depth == 0 {
            return Err(Error::InvalidConfig);
        }
        if self.memory_budget_bytes == 0
            || self.memory_budget_bytes < self.system_reserve_bytes
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    /// Resolve the requested backend against the reported capabilities. A GPU
    /// request without a GPU capability is an explicit unsupported error, never
    /// a silent CPU fallback.
    pub fn require_backend(
        &self,
        capabilities: ComputeCapabilities,
    ) -> Result<ComputeSelection> {
        match self.compute {
            ComputeSelection::Cpu => Ok(ComputeSelection::Cpu),
            ComputeSelection::RequireGpu if capabilities.gpu => {
                Ok(ComputeSelection::RequireGpu)
            }
            ComputeSelection::RequireGpu => Err(Error::UnsupportedBackend),
            ComputeSelection::Auto => Ok(if capabilities.gpu {
                ComputeSelection::RequireGpu
            } else {
                ComputeSelection::Cpu
            }),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    pub counters: BTreeMap<String, u64>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &str, value: u64) {
        self.counters.insert(name.to_owned(), value);
    }

    pub fn increment(&mut self, name: &str) {
        let counter = self.counters.entry(name.to_owned()).or_insert(0);
        *counter = counter.saturating_add(1);
    }

    pub fn get(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or(0)
    }
}

pub struct ExecContext<'a, F: ProtocolField> {
    pub compute: &'a mut dyn ComputeBackend<F>,
    pub storage: &'a dyn TensorStorageService,
    pub memory: &'a dyn MemoryGovernor,
    pub progress: &'a dyn ProgressSink,
    pub metrics: &'a mut Metrics,
    pub runtime: &'a RuntimeConfig,
    next_tensor_id: u64,
}

impl<'a, F: ProtocolField> ExecContext<'a, F> {
    pub fn new(
        compute: &'a mut dyn ComputeBackend<F>,
        storage: &'a dyn TensorStorageService,
        memory: &'a dyn MemoryGovernor,
        progress: &'a dyn ProgressSink,
        metrics: &'a mut Metrics,
        runtime: &'a RuntimeConfig,
    ) -> Self {
        Self {
            compute,
            storage,
            memory,
            progress,
            metrics,
            runtime,
            next_tensor_id: 0,
        }
    }
}

impl<'a, F: ProtocolField> ExecutionPort<F> for ExecContext<'a, F> {
    fn storage(&self) -> &dyn TensorStorageService {
        self.storage
    }

    fn memory(&self) -> &dyn MemoryGovernor {
        self.memory
    }

    fn progress(&self) -> &dyn ProgressSink {
        self.progress
    }

    fn execute(
        &mut self,
        job: &KernelJob<F>,
        inputs: &[TensorView],
        outputs: &[TensorSpec],
        workspace: &WorkspaceRequirement,
    ) -> Result<Vec<TensorView>> {
        self.runtime.validate()?;
        workspace.validate()?;
        let request = MemoryRequest::from_workspace(workspace);
        let permit = self.memory.acquire(&request)?;

        let mut input_handles = Vec::with_capacity(inputs.len());
        let mut upload_fences = Vec::with_capacity(inputs.len());
        for view in inputs {
            view.validate()?;
            let reader = self.storage.reader(view)?;
            let (_, byte_len) = view.element_range.byte_range(&view.spec)?;
            let byte_len = usize::try_from(byte_len).map_err(|_| Error::ResourceLimit)?;
            let mut bytes = vec![0u8; byte_len];
            let count = usize::try_from(view.element_range.len)
                .map_err(|_| Error::InvalidIndex)?;
            reader.read_range(view.element_range.start, count, &mut bytes)?;
            let handle = self.compute.allocate(
                &BufferSpec::new(view.spec.dtype, view.element_range.len),
                &permit,
            )?;
            upload_fences.push(self.compute.upload(handle, &bytes)?);
            input_handles.push(handle);
        }

        // TODO: published output buffers need a backend-owned permit lease for
        // their full lifetime; this foundation holds workspace accounting only
        // through the execute call.
        let mut output_handles = Vec::with_capacity(outputs.len());
        for spec in outputs {
            spec.validate()?;
            output_handles.push(self.compute.allocate(
                &BufferSpec::new(spec.dtype, spec.element_count()?),
                &permit,
            )?);
        }

        let fence = self
            .compute
            .submit(job, &input_handles, &output_handles, &upload_fences)?;
        self.compute.wait(fence)?;
        for handle in input_handles {
            self.compute.release(handle)?;
        }

        let mut views = Vec::with_capacity(outputs.len());
        for (spec, handle) in outputs.iter().zip(output_handles) {
            let count = spec.element_count()?;
            let tensor_id = TensorId(self.next_tensor_id);
            self.next_tensor_id = self
                .next_tensor_id
                .checked_add(1)
                .ok_or(Error::ResourceLimit)?;
            views.push(TensorView {
                tensor_id,
                spec: spec.clone(),
                storage: TensorStorageRef::Resident(handle),
                element_range: ElementRange::new(0, count),
            });
        }
        self.metrics.increment("kernel_submits");
        Ok(views)
    }
}

pub struct VerifyContext<'a> {
    pub policy: &'a VerifierPolicy,
    pub limits: &'a DecodeLimits,
    pub progress: &'a dyn ProgressSink,
    pub metrics: &'a mut Metrics,
}

impl<'a> VerifyContext<'a> {
    pub fn new(
        policy: &'a VerifierPolicy,
        limits: &'a DecodeLimits,
        progress: &'a dyn ProgressSink,
        metrics: &'a mut Metrics,
    ) -> Self {
        Self {
            policy,
            limits,
            progress,
            metrics,
        }
    }
}
