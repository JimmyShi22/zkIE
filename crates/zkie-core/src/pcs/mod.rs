pub mod batch_open;
pub mod cache;
pub mod committed;
#[cfg(feature = "cuda")]
pub mod cuda_buffer;
#[cfg(feature = "cuda")]
pub mod cuda_ffi;
#[cfg(feature = "cuda")]
pub mod dft_cuda;
#[cfg(feature = "cuda")]
pub mod merkle_cuda;
pub mod whir;

use crate::claim::EvalClaim;
use crate::common::codec::Canonical;
use crate::common::field::ProtocolField;
use crate::common::random::RandomSource;
use crate::common::transcript::ProtocolTranscript;
use crate::compute::{ExecutionPort, VerifierPolicy};
use crate::error::Result;
use crate::types::TensorView;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolynomialSpec {
    pub num_variables: u32,
    pub max_degree: u32,
}

/// Generic commitment interface. It is a building block, not the complete
/// request protocol: the session layer supplies the relation arguments,
/// hiding budget and the three chains.
pub trait CommitmentBackend<F: ProtocolField>: Send + Sync {
    type Params: Canonical + Send + Sync;
    type ProverKey: Send + Sync;
    type VerifierKey: Canonical + Send + Sync;
    type Commitment: Canonical + Clone + Send + Sync;
    type ProverData: Send;
    type OpeningProof: Canonical + Send;

    fn backend_id(&self) -> BackendId;

    fn validate_params(&self, params: &Self::Params, policy: &VerifierPolicy) -> Result<()>;

    fn prepare(
        &self,
        params: &Self::Params,
        exec: &mut dyn ExecutionPort<F>,
        rng: &mut dyn RandomSource,
    ) -> Result<(Self::ProverKey, Self::VerifierKey)>;

    fn commit(
        &self,
        pk: &Self::ProverKey,
        spec: &PolynomialSpec,
        values: &TensorView,
        exec: &mut dyn ExecutionPort<F>,
        rng: &mut dyn RandomSource,
        transcript: &mut dyn ProtocolTranscript<F>,
    ) -> Result<(Self::Commitment, Self::ProverData)>;

    fn open(
        &self,
        pk: &Self::ProverKey,
        data: &[&Self::ProverData],
        claims: &[EvalClaim<F>],
        exec: &mut dyn ExecutionPort<F>,
        rng: &mut dyn RandomSource,
        transcript: &mut dyn ProtocolTranscript<F>,
    ) -> Result<Self::OpeningProof>;

    fn verify(
        &self,
        vk: &Self::VerifierKey,
        commitments: &[Self::Commitment],
        claims: &[EvalClaim<F>],
        proof: &Self::OpeningProof,
        transcript: &mut dyn ProtocolTranscript<F>,
    ) -> Result<()>;
}
