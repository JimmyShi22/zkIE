//! Lookup argument interface.
//!
//! A lookup proof returns pending claims; it never closes the request on its
//! own. The table specification binds content, columns, key range, valid rows
//! and semantic version.

use std::marker::PhantomData;

use crate::claim::PendingClaims;
use crate::common::codec::Canonical;
use crate::common::field::ProtocolField;
use crate::common::random::RandomSource;
use crate::common::transcript::ProtocolTranscript;
use crate::compute::ExecutionPort;
use crate::error::{Error, Result};
use crate::oracle::OracleId;
use crate::types::{Digest, DType, TensorView};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableColumn {
    pub name: String,
    pub dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableSpec {
    pub digest: Digest,
    pub columns: Vec<TableColumn>,
    pub key_min: u64,
    pub key_max: u64,
    pub valid_rows: u64,
    pub semantic_version: u16,
}

impl TableSpec {
    pub fn validate(&self) -> Result<()> {
        if self.digest == Digest::ZERO || self.columns.is_empty() {
            return Err(Error::InvalidConfig);
        }
        if self.key_max < self.key_min {
            return Err(Error::InvalidIndex);
        }
        let span = self
            .key_max
            .checked_sub(self.key_min)
            .and_then(|span| span.checked_add(1))
            .ok_or(Error::InvalidIndex)?;
        if self.valid_rows > span {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupStatement<F: ProtocolField> {
    pub table_digest: Digest,
    pub query_oracle: OracleId,
    pub valid_rows: u64,
    pub padding_rows: u64,
    pub _field: PhantomData<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupWitness<F: ProtocolField> {
    pub queries: Vec<TensorView>,
    pub table: TensorView,
    pub _field: PhantomData<F>,
}

pub trait LookupArgument<F: ProtocolField>: Send + Sync {
    type Proof: Canonical + Send;

    fn validate_table(&self, table: &TableSpec) -> Result<()>;

    fn prove(
        &self,
        table: &TableSpec,
        statement: &LookupStatement<F>,
        witness: &LookupWitness<F>,
        exec: &mut dyn ExecutionPort<F>,
        rng: &mut dyn RandomSource,
        transcript: &mut dyn ProtocolTranscript<F>,
    ) -> Result<(Self::Proof, PendingClaims<F>)>;

    fn verify(
        &self,
        table: &TableSpec,
        statement: &LookupStatement<F>,
        proof: &Self::Proof,
        transcript: &mut dyn ProtocolTranscript<F>,
    ) -> Result<PendingClaims<F>>;
}
