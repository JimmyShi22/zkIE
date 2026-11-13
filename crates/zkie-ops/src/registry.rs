//! Shared operator registry plumbing: exact I/O, scratch records and a
//! storage-only execution port.
//!
//! Operators declare their math through [`PureForward`]; the blanket
//! [`Operation`] implementation reads input views through the storage service
//! and writes each output and witness as a sealed scratch tensor. No operator
//! creates its own thread pool or asset loader.

use std::marker::PhantomData;
use std::sync::Arc;

use rayon::prelude::*;

use zkie_core::common::field::ProtocolField;
use zkie_core::compute::{ExecutionPort, KernelJob, MemoryGovernor, MemoryRequest};
use zkie_core::error::{Error, Result};
use zkie_core::progress::{ProgressEvent, ProgressSink};
use zkie_core::storage::TensorStorageService;
use zkie_core::types::{
    DType, ElementRange, FixedSpec, TensorSpec, TensorView, WorkspaceRequirement,
};
use zkie_engine::primitives::rq;

use crate::op::{ForwardRecord, OpCode, OpPhase, Operation, OpStatement};
use zkie_core::claim::{PendingClaims, RelationId};
use zkie_core::oracle::OracleId;
use crate::proof::binding::{RelationReceipt, RelationProver, RelationVerifier, OracleRegistry};
use crate::proof::relations::{prove_op_relation, proof_input_values, verify_op_relation_with_config, OpRelationPublic, RelationConfig, RelationReceiptMessage};

/// The byte length of one I64 tensor with `count` elements.
pub fn i64_bytes(count: usize) -> Result<u64> {
    let count = u64::try_from(count).map_err(|_| Error::ResourceLimit)?;
    count.checked_mul(8).ok_or(Error::ResourceLimit)
}

/// Read a whole I64 view through the storage service. Reads are ranged; a
/// missing dtype is an explicit error.
pub fn read_i64_view(storage: &dyn TensorStorageService, view: &TensorView) -> Result<Vec<i64>> {
    view.validate()?;
    if !matches!(view.spec.dtype, DType::I64) {
        return Err(Error::UnsupportedPrimitive);
    }
    let count = usize::try_from(view.element_range.len).map_err(|_| Error::ResourceLimit)?;
    let reader = storage.reader(view)?;
    let mut bytes = vec![0u8; count.checked_mul(8).ok_or(Error::ResourceLimit)?];
    reader.read_range(view.element_range.start, count, &mut bytes)?;
    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8) {
        values.push(i64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?));
    }
    Ok(values)
}

pub fn encode_i64(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Write values as a sealed scratch tensor. The caller's permit covers exactly
/// this tensor's bytes and is released after the write.
pub fn write_i64_scratch(
    storage: &dyn TensorStorageService,
    memory: &dyn MemoryGovernor,
    spec: &TensorSpec,
    values: &[i64],
) -> Result<TensorView> {
    spec.validate()?;
    if !matches!(spec.dtype, DType::I64) {
        return Err(Error::UnsupportedPrimitive);
    }
    if spec.element_count()? != values.len() as u64 {
        return Err(Error::ShapeMismatch);
    }
    let bytes = encode_i64(values);
    let permit = memory.acquire(&MemoryRequest {
        resident_bytes: 0,
        mapped_window_bytes: 0,
        pinned_bytes: 0,
        metadata_bytes: 0,
        scratch_disk_bytes: bytes.len() as u64,
    })?;
    let id = storage.create_scratch(spec, &permit)?;
    storage.write_scratch(id, ElementRange::new(0, values.len() as u64), &bytes)?;
    let view = storage.seal_scratch(id)?;
    drop(permit);
    Ok(view)
}

/// One operator's pure forward result. Raw accumulators are signed 128-bit
/// values and are stored as two I64 words per element.
#[derive(Clone, Debug, Default)]
pub struct KernelRecord {
    pub outputs: Vec<Vec<i64>>,
    pub raw_accumulators: Vec<Vec<i128>>,
    pub quotients: Vec<Vec<i64>>,
    pub range_helpers: Vec<Vec<i64>>,
    pub auxiliaries: Vec<Vec<i64>>,
}

impl KernelRecord {
    pub fn with_output(output: Vec<i64>) -> Self {
        Self {
            outputs: vec![output],
            ..Self::default()
        }
    }
}

/// The mathematical part of an operator. Implementations are pure and never
/// touch storage; the blanket `Operation` adapter handles I/O.
pub trait PureForward: Send + Sync {
    fn opcode(&self) -> OpCode;
    fn infer_specs(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>>;
    fn forward_values(&self, inputs: &[Vec<i64>]) -> Result<KernelRecord>;
}

fn output_specs(op: &dyn PureForward, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
    op.infer_specs(inputs)
}

impl<F: ProtocolField, T: PureForward + RelationConfig> Operation<F> for T {
    fn opcode(&self) -> OpCode {
        PureForward::opcode(self)
    }

    fn infer(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        output_specs(self, inputs)
    }

    fn workspace(
        &self,
        _phase: OpPhase,
        inputs: &[TensorSpec],
    ) -> Result<WorkspaceRequirement> {
        let mut bytes = 0u64;
        for spec in inputs {
            bytes = bytes
                .checked_add(spec.byte_len()?)
                .ok_or(Error::ResourceLimit)?;
        }
        for spec in PureForward::infer_specs(self, inputs)? {
            bytes = bytes
                .checked_add(spec.byte_len()?)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(WorkspaceRequirement {
            minimum_resident_bytes: 0,
            selected_resident_bytes: 0,
            mapped_window_bytes: 0,
            scratch_disk_bytes: bytes,
        })
    }

    fn forward(
        &self,
        ctx: &mut dyn ExecutionPort<F>,
        inputs: &[TensorView],
    ) -> Result<ForwardRecord> {
        let values: Vec<Vec<i64>> = inputs
            .iter()
            .map(|view| read_i64_view(ctx.storage(), view))
            .collect::<Result<_>>()?;
        let output_specs = self.infer_specs(&inputs.iter().map(|view| view.spec.clone()).collect::<Vec<_>>())?;
        let record = self.forward_values(&values)?;
        if record.outputs.len() != output_specs.len() {
            return Err(Error::ShapeMismatch);
        }
        let mut header = ForwardRecord::default();
        header.inputs = inputs.to_vec();
        for (spec, values) in output_specs.iter().zip(&record.outputs) {
            header
                .outputs
                .push(write_i64_scratch(ctx.storage(), ctx.memory(), spec, values)?);
        }
        for raw in &record.raw_accumulators {
            let mut words = Vec::with_capacity(raw.len() * 2);
            for value in raw {
                words.push(*value as i64);
                words.push((*value >> 64) as i64);
            }
            let spec = TensorSpec::new(DType::I64, vec![words.len() as u64]);
            header
                .raw_accumulators
                .push(write_i64_scratch(ctx.storage(), ctx.memory(), &spec, &words)?);
        }
        for values in &record.quotients {
            let spec = TensorSpec::new(DType::I64, vec![values.len() as u64]);
            header
                .quotients
                .push(write_i64_scratch(ctx.storage(), ctx.memory(), &spec, values)?);
        }
        for values in &record.range_helpers {
            let spec = TensorSpec::new(DType::I64, vec![values.len() as u64]);
            header
                .range_helpers
                .push(write_i64_scratch(ctx.storage(), ctx.memory(), &spec, values)?);
        }
        for values in &record.auxiliaries {
            let spec = TensorSpec::new(DType::I64, vec![values.len() as u64]);
            header
                .auxiliaries
                .push(write_i64_scratch(ctx.storage(), ctx.memory(), &spec, values)?);
        }
        Ok(header)
    }

    fn prove(
        &self,
        ctx: &mut dyn ExecutionPort<F>,
        session: &mut dyn RelationProver<F>,
        statement: &OpStatement,
        record: &ForwardRecord,
    ) -> Result<RelationReceipt> {
        let mut inputs = Vec::with_capacity(record.inputs.len());
        for view in &record.inputs {
            inputs.push(read_i64_view(ctx.storage(), view)?);
        }
        let config = self.relation_config(&inputs)?;
        let (public, proof) = prove_op_relation(self, &inputs, &config)?;
        let mut roots = Vec::new();
        for id in public.registry.ids() {
            let entry = public.registry.get(id).ok_or(Error::ProverStateMissing)?;
            roots.push((entry.root, entry.start, entry.count));
        }
        let message = bincode::serialize(&RelationReceiptMessage {
            opcode: self.opcode() as u16,
            config,
            roots,
            external_inputs: Vec::new(),
            proof,
        })
        .map_err(|_| Error::InvalidEncoding)?;
        session.register_claims(PendingClaims::new())?;
        let receipt = RelationReceipt {
            relation: RelationId(statement.op_id.0 as u64),
            message,
        };
        session.submit_receipt(&receipt)?;
        Ok(receipt)
    }

    fn verify(
        &self,
        session: &mut dyn RelationVerifier<F>,
        statement: &OpStatement,
        receipt: &RelationReceipt,
    ) -> Result<PendingClaims<F>> {
        if statement.opcode != self.opcode() || receipt.relation.0 != statement.op_id.0 as u64 {
            return Err(Error::StatementMismatch);
        }
        let message: RelationReceiptMessage =
            bincode::deserialize(&receipt.message).map_err(|_| Error::InvalidEncoding)?;
        if message.opcode != self.opcode() as u16 {
            return Err(Error::StatementMismatch);
        }
        let mut registry = OracleRegistry::new();
        for (root, start, count) in &message.roots {
            registry.insert(*root, *start, *count)?;
        }
        let input_count = message.proof.lengths.inputs.len();
        let output_count = message.proof.lengths.outputs.len();
        let input_oracles = (0..input_count)
            .map(|index| OracleId(index as u64))
            .collect::<Vec<_>>();
        let output_oracles = (input_count..input_count + output_count)
            .map(|index| OracleId(index as u64))
            .collect::<Vec<_>>();
        let public = OpRelationPublic {
            opcode: self.opcode(),
            config: message.config.clone(),
            registry,
            input_oracles,
            output_oracles,
        };
        let inputs = proof_input_values(&message.proof)?;
        let expected = self.relation_config(&inputs)?;
        verify_op_relation_with_config(&public, &message.proof, &expected)?;
        session.register_claims(PendingClaims::new())?;
        Ok(PendingClaims::new())
    }
}

/// Exact requantization: `rq(value, shift)` with mathematical floor.
pub fn requantize(value: i128, shift: u32) -> Result<i128> {
    rq(value, shift)
}

/// Split a signed 128-bit value into two I64 words.
pub fn wide_words(value: i128) -> [i64; 2] {
    [value as i64, (value >> 64) as i64]
}

/// Recombine two I64 words into a signed 128-bit value.
pub fn wide_from_words(low: i64, high: i64) -> i128 {
    ((high as i128) << 64) | (low as u64 as i128)
}

pub fn fixed_spec(scale: i32, bound: u128) -> FixedSpec {
    FixedSpec::new(scale, bound, zkie_core::types::RoundRule::Floor)
}

/// A storage-only port for forward tests and for callers that supply their own
/// compute backend elsewhere. `execute` is intentionally unsupported here.
pub struct StorageOnlyPort<'a, F: ProtocolField> {
    pub storage: &'a dyn TensorStorageService,
    pub memory: &'a dyn MemoryGovernor,
    pub progress: &'a dyn ProgressSink,
    marker: PhantomData<F>,
}

impl<'a, F: ProtocolField> StorageOnlyPort<'a, F> {
    pub fn new(
        storage: &'a dyn TensorStorageService,
        memory: &'a dyn MemoryGovernor,
        progress: &'a dyn ProgressSink,
    ) -> Self {
        Self {
            storage,
            memory,
            progress,
            marker: PhantomData,
        }
    }
}

impl<'a, F: ProtocolField> ExecutionPort<F> for StorageOnlyPort<'a, F> {
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
        _job: &KernelJob<F>,
        _inputs: &[TensorView],
        _outputs: &[TensorSpec],
        _workspace: &WorkspaceRequirement,
    ) -> Result<Vec<TensorView>> {
        Err(Error::UnsupportedKernel)
    }
}

pub struct NullProgress;

impl ProgressSink for NullProgress {
    fn emit(&self, _event: ProgressEvent) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// A scratch directory that removes itself.
pub struct TempScratch {
    pub path: std::path::PathBuf,
}

impl TempScratch {
    pub fn new(tag: &str) -> Result<Self> {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "zkie-op-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&path).map_err(|_| Error::Io)?;
        Ok(Self { path })
    }
}

impl Drop for TempScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Shared scratch storage for a test or a call site that has no engine.
pub fn scratch_storage(dir: &std::path::Path) -> Result<Arc<dyn TensorStorageService>> {
    let storage = zkie_engine::storage::EngineStorage::new(dir)?;
    Ok(Arc::new(storage))
}

pub fn spec_1d(count: usize) -> TensorSpec {
    TensorSpec::new(DType::I64, vec![count as u64])
}

pub fn io_count(spec: &TensorSpec) -> Result<usize> {
    usize::try_from(spec.element_count()?).map_err(|_| Error::ResourceLimit)
}

pub fn require_len(values: &[i64], length: usize) -> Result<()> {
    if values.len() != length {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}

pub fn to_i64_checked(value: i128) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::BoundViolation)
}

pub fn round_div_half_up(value: i128, divisor: i128) -> Result<i128> {
    if divisor <= 0 {
        return if divisor == 0 {
            Err(Error::DivisionByZero)
        } else {
            Err(Error::InvalidConfig)
        };
    }
    let quotient = value.div_euclid(divisor);
    let remainder = value.rem_euclid(divisor);
    Ok(if remainder * 2 >= divisor {
        quotient + 1
    } else {
        quotient
    })
}

pub fn floor_div(value: i128, divisor: i128) -> Result<i128> {
    if divisor <= 0 {
        return if divisor == 0 {
            Err(Error::DivisionByZero)
        } else {
            Err(Error::InvalidConfig)
        };
    }
    Ok(value.div_euclid(divisor))
}

pub fn exp_value(table: &[i64], distance: i64) -> Result<i64> {
    if distance < 0 || distance as usize >= table.len() {
        return Err(Error::BoundViolation);
    }
    Ok(table[distance as usize])
}

/// Scale of the profile rsqrt coarse rounding step.
pub const RSQRT_SHIFT: i64 = 1 << 18;
/// Half of the profile rsqrt coarse rounding step, the round-half-up offset.
pub const RSQRT_HALF: i64 = 1 << 17;
/// First fine-table index bound of the profile rsqrt segmentation.
pub const RSQRT_FINE: i64 = 1 << 20;
/// Coarse segment span of the profile rsqrt segmentation.
pub const RSQRT_COARSE: i64 = 1 << 22;
/// End of the coarse segment of the profile rsqrt segmentation.
pub const RSQRT_COARSE_END: i64 = RSQRT_FINE + RSQRT_COARSE * (1 << 8);

/// Piecewise rsqrt table index shared by the frozen norm profiles.
pub fn profile_rsqrt_index(var: i64) -> Result<u64> {
    let raw = round_div_half_up(var as i128, RSQRT_SHIFT as i128)?;
    let raw = i64::try_from(raw).map_err(|_| Error::BoundViolation)?;
    if raw < RSQRT_FINE {
        Ok(raw.max(0) as u64)
    } else if raw < RSQRT_COARSE_END {
        Ok((RSQRT_FINE + ((raw - RSQRT_FINE) >> 8)) as u64)
    } else {
        Ok((RSQRT_FINE + RSQRT_COARSE + ((raw - RSQRT_COARSE_END) >> 16)) as u64)
    }
}

/// Profile softmax lookup: index = (score - max) + offset, where the frozen
/// tables use offset = table length (2^21), clamped to the last entry.
pub fn exp_profile_value(table: &[i64], distance: i64) -> Result<i64> {
    if distance < 0 || table.is_empty() {
        return Err(Error::BoundViolation);
    }
    let index = (table.len() as i64)
        .checked_sub(distance)
        .ok_or(Error::BoundViolation)?;
    let index = index.clamp(0, table.len() as i64 - 1) as usize;
    Ok(table[index])
}

pub fn matmul_raw(a: &[i64], b: &[i64], m: usize, n: usize, k: usize) -> Result<Vec<i128>> {
    let _profile = zkie_core::profile::Scope::kernel(zkie_core::profile::Kernel::Matmul);
    require_len(a, m.checked_mul(k).ok_or(Error::ResourceLimit)?)?;
    require_len(b, k.checked_mul(n).ok_or(Error::ResourceLimit)?)?;
    let mut out = vec![0i128; m.checked_mul(n).ok_or(Error::ResourceLimit)?];
    if out.is_empty() {
        return Ok(out);
    }
    out.par_chunks_mut(n).enumerate().try_for_each(|(row, chunk)| -> Result<()> {
        for inner in 0..k {
            let left = a[row * k + inner] as i128;
            if left == 0 {
                continue;
            }
            for column in 0..n {
                let term = left
                    .checked_mul(b[inner * n + column] as i128)
                    .ok_or(Error::BoundViolation)?;
                chunk[column] = chunk[column]
                    .checked_add(term)
                    .ok_or(Error::BoundViolation)?;
            }
        }
        Ok(())
    })?;
    Ok(out)
}

pub fn isqrt_u128(value: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 1u128 << 64;
    while low < high {
        let middle = low + (high - low + 1) / 2;
        match middle.checked_mul(middle) {
            Some(square) if square <= value => low = middle,
            _ => high = middle - 1,
        }
    }
    low
}
