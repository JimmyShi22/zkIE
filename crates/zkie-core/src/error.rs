//! Error classification shared by every layer.
//!
//! Errors are deliberately coarse: callers branch on the variant, not on a
//! free-form message. A recoverable compute retry must reuse the authenticated
//! inputs and must not consume protocol randomness.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    UnsupportedPrimitive,
    UnsupportedKernel,
    UnsupportedBackend,
    InvalidConfig,
    ShapeMismatch,
    ScaleMismatch,
    InvalidIndex,
    InvalidHandle,
    BoundViolation,
    DivisionByZero,
    InvalidEncoding,
    UnsupportedVersion,
    StatementMismatch,
    InvalidProof,
    UnclosedClaims,
    InvalidSessionState,
    ResourceLimit,
    DiskLimit,
    DeviceUnavailable,
    DeviceOutOfMemory,
    DeviceLost,
    SynchronizationFailed,
    EntropyUnavailable,
    CacheMismatch,
    AssetMismatch,
    MissingCommitArtifact,
    IncompleteArtifact,
    ArtifactVersionMismatch,
    ArtifactIntegrityMismatch,
    ArtifactAlreadyExists,
    ProverStateMissing,
    LeafBudgetExhausted,
    StateLocked,
    StateRollbackSuspected,
    ObservabilityIo,
    Cancelled,
    Io,
}

impl Error {
    /// Stable machine-readable code used by progress events and reports.
    pub const fn code(self) -> &'static str {
        match self {
            Error::UnsupportedPrimitive => "unsupported_primitive",
            Error::UnsupportedKernel => "unsupported_kernel",
            Error::UnsupportedBackend => "unsupported_backend",
            Error::InvalidConfig => "invalid_config",
            Error::ShapeMismatch => "shape_mismatch",
            Error::ScaleMismatch => "scale_mismatch",
            Error::InvalidIndex => "invalid_index",
            Error::InvalidHandle => "invalid_handle",
            Error::BoundViolation => "bound_violation",
            Error::DivisionByZero => "division_by_zero",
            Error::InvalidEncoding => "invalid_encoding",
            Error::UnsupportedVersion => "unsupported_version",
            Error::StatementMismatch => "statement_mismatch",
            Error::InvalidProof => "invalid_proof",
            Error::UnclosedClaims => "unclosed_claims",
            Error::InvalidSessionState => "invalid_session_state",
            Error::ResourceLimit => "resource_limit",
            Error::DiskLimit => "disk_limit",
            Error::DeviceUnavailable => "device_unavailable",
            Error::DeviceOutOfMemory => "device_out_of_memory",
            Error::DeviceLost => "device_lost",
            Error::SynchronizationFailed => "synchronization_failed",
            Error::EntropyUnavailable => "entropy_unavailable",
            Error::CacheMismatch => "cache_mismatch",
            Error::AssetMismatch => "asset_mismatch",
            Error::MissingCommitArtifact => "missing_commit_artifact",
            Error::IncompleteArtifact => "incomplete_artifact",
            Error::ArtifactVersionMismatch => "artifact_version_mismatch",
            Error::ArtifactIntegrityMismatch => "artifact_integrity_mismatch",
            Error::ArtifactAlreadyExists => "artifact_already_exists",
            Error::ProverStateMissing => "prover_state_missing",
            Error::LeafBudgetExhausted => "leaf_budget_exhausted",
            Error::StateLocked => "state_locked",
            Error::StateRollbackSuspected => "state_rollback_suspected",
            Error::ObservabilityIo => "observability_io",
            Error::Cancelled => "cancelled",
            Error::Io => "io",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
