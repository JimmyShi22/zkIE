//! Operator interface.
//!
//! An operator declares its shape inference, workspace requirement, forward
//! record and relation submission. It never creates a thread pool, a transcript,
//! a commitment backend or its own proof lifecycle.

use zkie_core::claim::PendingClaims;
use zkie_core::common::field::ProtocolField;
use zkie_core::compute::ExecutionPort;
use zkie_core::error::{Error, Result};
use zkie_core::types::{Digest, TensorId, TensorSpec, TensorView, WorkspaceRequirement};

use crate::proof::binding::{RelationProver, RelationReceipt, RelationVerifier};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpCode {
    Layout,
    Requantize,
    AffineScale,
    Relu,
    MatMul,
    Projection,
    Add,
    TableLookup,
    Softmax,
    LayerNorm,
    RmsNorm,
    RopeProjection,
    TopK,
    Argmax,
    RouteGate,
    Scatter,
    Gather,
    BatchedMatMul,
    GatherMatMul,
    IndexGather,
    Pad,
    Hadamard,
    Range,
    DivRound,
    IntSqrt,
    GatedFfn,
    MoeDown,
    KvCache,
    Patchify,
    RegressionHead,
    CausalAttention,
    ProfileLayerNorm,
    TableIndex,
    EmbeddingIndex,
    ProfileRmsNorm,
    RowEmbeddingIndex,
    ProfileRope,
    ProfileStandardize,
    SoftmaxRows,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpPhase {
    Forward,
    Prove,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    Output,
    RawAccumulator,
    Quotient,
    RangeHelper,
    Auxiliary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpStatement {
    pub op_id: OpId,
    pub opcode: OpCode,
    pub semantic_version: u16,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub parameters_digest: Digest,
}

impl OpStatement {
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() && self.outputs.is_empty() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// Everything the forward pass must retain for a later relation proof.
/// Views may point at resident buffers, mapped files or scratch files.
#[derive(Clone, Debug, Default)]
pub struct ForwardRecord {
    pub inputs: Vec<TensorView>,
    pub outputs: Vec<TensorView>,
    pub raw_accumulators: Vec<TensorView>,
    pub quotients: Vec<TensorView>,
    pub range_helpers: Vec<TensorView>,
    pub auxiliaries: Vec<TensorView>,
}

impl ForwardRecord {
    pub fn new(outputs: Vec<TensorView>) -> Self {
        Self {
            outputs,
            ..Self::default()
        }
    }

    pub fn views(&self) -> impl Iterator<Item = &TensorView> {
        self.inputs
            .iter()
            .chain(self.outputs.iter())
            .chain(self.raw_accumulators.iter())
            .chain(self.quotients.iter())
            .chain(self.range_helpers.iter())
            .chain(self.auxiliaries.iter())
    }

    pub fn views_for(&self, role: TensorRole) -> &[TensorView] {
        match role {
            TensorRole::Output => &self.outputs,
            TensorRole::RawAccumulator => &self.raw_accumulators,
            TensorRole::Quotient => &self.quotients,
            TensorRole::RangeHelper => &self.range_helpers,
            TensorRole::Auxiliary => &self.auxiliaries,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for view in self.views() {
            view.validate()?;
        }
        Ok(())
    }
}

pub trait Operation<F: ProtocolField>: Send + Sync {
    fn opcode(&self) -> OpCode;

    fn semantic_version(&self) -> u16 {
        1
    }

    fn infer(&self, inputs: &[TensorSpec]) -> Result<Vec<TensorSpec>>;

    fn workspace(
        &self,
        phase: OpPhase,
        inputs: &[TensorSpec],
    ) -> Result<WorkspaceRequirement>;

    fn forward(
        &self,
        ctx: &mut dyn ExecutionPort<F>,
        inputs: &[TensorView],
    ) -> Result<ForwardRecord>;

    /// Submit the op relation to the session. The session owns claim closure
    /// and the proof lifecycle; this default is for operators that do not yet
    /// implement a relation.
    fn prove(
        &self,
        _ctx: &mut dyn ExecutionPort<F>,
        _session: &mut dyn RelationProver<F>,
        _statement: &OpStatement,
        _record: &ForwardRecord,
    ) -> Result<RelationReceipt> {
        Err(Error::UnsupportedPrimitive)
    }

    /// Verify the op relation and return pending claims to the session.
    fn verify(
        &self,
        _session: &mut dyn RelationVerifier<F>,
        _statement: &OpStatement,
        _receipt: &RelationReceipt,
    ) -> Result<PendingClaims<F>> {
        Err(Error::UnsupportedPrimitive)
    }
}

pub fn opcode_from_u16(value: u16) -> zkie_core::error::Result<OpCode> {
    let opcode = match value {
        0 => OpCode::Layout,
        1 => OpCode::Requantize,
        2 => OpCode::AffineScale,
        3 => OpCode::Relu,
        4 => OpCode::MatMul,
        5 => OpCode::Projection,
        6 => OpCode::Add,
        7 => OpCode::TableLookup,
        8 => OpCode::Softmax,
        9 => OpCode::LayerNorm,
        10 => OpCode::RmsNorm,
        11 => OpCode::RopeProjection,
        12 => OpCode::TopK,
        13 => OpCode::Argmax,
        14 => OpCode::RouteGate,
        15 => OpCode::Scatter,
        16 => OpCode::Gather,
        17 => OpCode::BatchedMatMul,
        18 => OpCode::GatherMatMul,
        19 => OpCode::IndexGather,
        20 => OpCode::Pad,
        21 => OpCode::Hadamard,
        22 => OpCode::Range,
        23 => OpCode::DivRound,
        24 => OpCode::IntSqrt,
        25 => OpCode::GatedFfn,
        26 => OpCode::MoeDown,
        27 => OpCode::KvCache,
        28 => OpCode::Patchify,
        29 => OpCode::RegressionHead,
        30 => OpCode::CausalAttention,
        31 => OpCode::ProfileLayerNorm,
        32 => OpCode::TableIndex,
        33 => OpCode::EmbeddingIndex,
        34 => OpCode::ProfileRmsNorm,
        35 => OpCode::RowEmbeddingIndex,
        36 => OpCode::ProfileRope,
        37 => OpCode::ProfileStandardize,
        38 => OpCode::SoftmaxRows,
        _ => return Err(zkie_core::error::Error::InvalidEncoding),
    };
    Ok(opcode)
}
