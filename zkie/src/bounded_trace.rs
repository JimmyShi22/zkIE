//! Bounded decoding for the production shared trace.
//!
//! The wire format is materialized by one streaming decoder. Every declared
//! collection is charged before its backing allocation and before its elements
//! are visited. The decoder never rewinds or buffers the complete proof;
//! reads go through a fixed-size buffer.

use std::cell::Cell;
use std::fmt;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use bincode::Options;
use serde::de::{DeserializeSeed, Deserializer, Error as DeError, SeqAccess, Visitor};
use serde::Deserialize;

use zkie_core::error::{Error, Result};
use zkie_core::oracle::OracleId;
use zkie_core::types::Digest;
use zkie_engine::pcs::request::link::{LeafRecord, LinkProof};
use zkie_ops::proof::binding::OracleOpening;
use zkie_ops::proof::model_commitment::{AssetOpening, RowOpening, SharedTrace, SharedTraceStep};
use zkie_ops::proof::relations::{OpRelationProof, RelationLengths, RelationReceiptMessage};

#[derive(Clone, Copy, Debug)]
struct BudgetState {
    limit: usize,
    per_allocation_limit: usize,
    charged: usize,
    largest_allocation: usize,
    allocations: usize,
    exceeded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationUsage {
    charged_bytes: usize,
    largest_allocation_bytes: usize,
    allocations: usize,
}

thread_local! {
    static PREFLIGHT_BUDGET: Cell<Option<BudgetState>> = const { Cell::new(None) };
}

struct BudgetGuard;

impl BudgetGuard {
    fn enter(limit: usize) -> Result<Self> {
        let occupied = PREFLIGHT_BUDGET.with(|slot| {
            slot.replace(Some(BudgetState {
                limit,
                per_allocation_limit: limit,
                charged: 0,
                largest_allocation: 0,
                allocations: 0,
                exceeded: false,
            }))
        });
        if occupied.is_some() {
            PREFLIGHT_BUDGET.with(|slot| slot.set(occupied));
            return Err(Error::InvalidSessionState);
        }
        Ok(Self)
    }

    fn state(&self) -> BudgetState {
        PREFLIGHT_BUDGET
            .with(Cell::get)
            .expect("bounded trace budget is active")
    }

    fn usage(&self) -> AllocationUsage {
        let state = self.state();
        AllocationUsage {
            charged_bytes: state.charged,
            largest_allocation_bytes: state.largest_allocation,
            allocations: state.allocations,
        }
    }
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        PREFLIGHT_BUDGET.with(|slot| slot.set(None));
    }
}

fn mark_exceeded() {
    PREFLIGHT_BUDGET.with(|slot| {
        if let Some(mut state) = slot.get() {
            state.exceeded = true;
            slot.set(Some(state));
        }
    });
}

fn charge_allocation(bytes: usize) -> bool {
    PREFLIGHT_BUDGET.with(|slot| {
        let Some(mut state) = slot.get() else {
            return false;
        };
        match state.charged.checked_add(bytes) {
            Some(charged) if bytes <= state.per_allocation_limit && charged <= state.limit => {
                state.charged = charged;
                if bytes != 0 {
                    state.largest_allocation = state.largest_allocation.max(bytes);
                    state.allocations = state.allocations.saturating_add(1);
                }
            }
            _ => state.exceeded = true,
        }
        slot.set(Some(state));
        !state.exceeded
    })
}

fn reserve_vector<T, E>(elements: usize) -> std::result::Result<Vec<T>, E>
where
    E: DeError,
{
    let requested = elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| E::custom("collection allocation overflow"))?;
    if !charge_allocation(requested) {
        return Err(E::custom("collection allocation limit"));
    }
    let mut values = Vec::new();
    if values.try_reserve_exact(elements).is_err() {
        mark_exceeded();
        return Err(E::custom("collection allocation failed"));
    }
    let actual = values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| {
            mark_exceeded();
            E::custom("collection capacity overflow")
        })?;
    if actual > requested && !charge_allocation(actual - requested) {
        return Err(E::custom("collection allocation limit"));
    }
    Ok(values)
}

struct ValueSeed<T>(PhantomData<T>);

impl<T> ValueSeed<T> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Clone for ValueSeed<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ValueSeed<T> {}

impl<'de, T> DeserializeSeed<'de> for ValueSeed<T>
where
    T: Deserialize<'de>,
{
    type Value = T;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer)
    }
}

#[derive(Clone, Copy)]
struct VecSeed<S> {
    element: S,
}

impl<S> VecSeed<S> {
    const fn new(element: S) -> Self {
        Self { element }
    }
}

impl<'de, S> DeserializeSeed<'de> for VecSeed<S>
where
    S: DeserializeSeed<'de> + Copy,
{
    type Value = Vec<S::Value>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(VecVisitor {
            element: self.element,
        })
    }
}

struct VecVisitor<S> {
    element: S,
}

impl<'de, S> Visitor<'de> for VecVisitor<S>
where
    S: DeserializeSeed<'de> + Copy,
{
    type Value = Vec<S::Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a length-prefixed bounded collection")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let elements = sequence
            .size_hint()
            .ok_or_else(|| A::Error::custom("collection length is unavailable"))?;
        let mut values = reserve_vector::<S::Value, A::Error>(elements)?;
        for _ in 0..elements {
            let value = sequence
                .next_element_seed(self.element)?
                .ok_or_else(|| A::Error::custom("truncated collection"))?;
            values.push(value);
        }
        Ok(values)
    }
}

fn next_field<'de, A, S>(
    sequence: &mut A,
    seed: S,
    name: &'static str,
) -> std::result::Result<S::Value, A::Error>
where
    A: SeqAccess<'de>,
    S: DeserializeSeed<'de>,
{
    sequence
        .next_element_seed(seed)?
        .ok_or_else(|| A::Error::custom(format_args!("missing {name}")))
}

#[derive(Clone, Copy)]
struct RowOpeningSeed;

impl<'de> DeserializeSeed<'de> for RowOpeningSeed {
    type Value = RowOpening;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("RowOpening", &["row", "values", "path"], RowOpeningVisitor)
    }
}

struct RowOpeningVisitor;

impl<'de> Visitor<'de> for RowOpeningVisitor {
    type Value = RowOpening;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a row opening")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(RowOpening {
            row: next_field(&mut sequence, ValueSeed::<u64>::new(), "row")?,
            values: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<u64>::new()),
                "values",
            )?,
            path: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<Digest>::new()),
                "path",
            )?,
        })
    }
}

#[derive(Clone, Copy)]
struct LinkProofSeed;

impl<'de> DeserializeSeed<'de> for LinkProofSeed {
    type Value = LinkProof;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "LinkProof",
            &["anchor_id", "registry_index", "record", "binding", "path"],
            LinkProofVisitor,
        )
    }
}

struct LinkProofVisitor;

impl<'de> Visitor<'de> for LinkProofVisitor {
    type Value = LinkProof;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a link proof")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(LinkProof {
            anchor_id: next_field(&mut sequence, ValueSeed::<Digest>::new(), "anchor_id")?,
            registry_index: next_field(&mut sequence, ValueSeed::<u64>::new(), "registry_index")?,
            record: next_field(&mut sequence, ValueSeed::<LeafRecord>::new(), "record")?,
            binding: next_field(&mut sequence, ValueSeed::<Digest>::new(), "binding")?,
            path: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<Digest>::new()),
                "path",
            )?,
        })
    }
}

#[derive(Clone, Copy)]
struct AssetOpeningSeed;

impl<'de> DeserializeSeed<'de> for AssetOpeningSeed {
    type Value = AssetOpening;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "AssetOpening",
            &["asset_id", "link", "rows", "start", "root", "row_count"],
            AssetOpeningVisitor,
        )
    }
}

struct AssetOpeningVisitor;

impl<'de> Visitor<'de> for AssetOpeningVisitor {
    type Value = AssetOpening;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an asset opening")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(AssetOpening {
            asset_id: next_field(&mut sequence, ValueSeed::<u64>::new(), "asset_id")?,
            link: next_field(&mut sequence, LinkProofSeed, "link")?,
            rows: next_field(&mut sequence, VecSeed::new(RowOpeningSeed), "rows")?,
            start: next_field(&mut sequence, ValueSeed::<Digest>::new(), "start")?,
            root: next_field(&mut sequence, ValueSeed::<Digest>::new(), "root")?,
            row_count: next_field(&mut sequence, ValueSeed::<u64>::new(), "row_count")?,
        })
    }
}

#[derive(Clone, Copy)]
struct OracleOpeningSeed;

impl<'de> DeserializeSeed<'de> for OracleOpeningSeed {
    type Value = OracleOpening;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "OracleOpening",
            &["oracle", "index", "values", "path"],
            OracleOpeningVisitor,
        )
    }
}

struct OracleOpeningVisitor;

impl<'de> Visitor<'de> for OracleOpeningVisitor {
    type Value = OracleOpening;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an oracle opening")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(OracleOpening {
            oracle: next_field(&mut sequence, ValueSeed::<OracleId>::new(), "oracle")?,
            index: next_field(&mut sequence, ValueSeed::<u64>::new(), "index")?,
            values: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<u64>::new()),
                "values",
            )?,
            path: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<Digest>::new()),
                "path",
            )?,
        })
    }
}

#[derive(Clone, Copy)]
struct RelationLengthsSeed;

impl<'de> DeserializeSeed<'de> for RelationLengthsSeed {
    type Value = RelationLengths;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "RelationLengths",
            &["inputs", "outputs", "raw", "quotients", "helpers", "aux"],
            RelationLengthsVisitor,
        )
    }
}

struct RelationLengthsVisitor;

impl<'de> Visitor<'de> for RelationLengthsVisitor {
    type Value = RelationLengths;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("relation lengths")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let seed = VecSeed::new(ValueSeed::<usize>::new());
        Ok(RelationLengths {
            inputs: next_field(&mut sequence, seed, "inputs")?,
            outputs: next_field(&mut sequence, seed, "outputs")?,
            raw: next_field(&mut sequence, seed, "raw")?,
            quotients: next_field(&mut sequence, seed, "quotients")?,
            helpers: next_field(&mut sequence, seed, "helpers")?,
            aux: next_field(&mut sequence, seed, "aux")?,
        })
    }
}

#[derive(Clone, Copy)]
struct RelationProofSeed;

impl<'de> DeserializeSeed<'de> for RelationProofSeed {
    type Value = OpRelationProof;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "OpRelationProof",
            &[
                "lengths",
                "inputs",
                "outputs",
                "raw",
                "quotients",
                "helpers",
                "aux",
            ],
            RelationProofVisitor,
        )
    }
}

struct RelationProofVisitor;

impl<'de> Visitor<'de> for RelationProofVisitor {
    type Value = OpRelationProof;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a relation proof")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let openings = VecSeed::new(VecSeed::new(OracleOpeningSeed));
        Ok(OpRelationProof {
            lengths: next_field(&mut sequence, RelationLengthsSeed, "lengths")?,
            inputs: next_field(&mut sequence, openings, "inputs")?,
            outputs: next_field(&mut sequence, openings, "outputs")?,
            raw: next_field(&mut sequence, openings, "raw")?,
            quotients: next_field(&mut sequence, openings, "quotients")?,
            helpers: next_field(&mut sequence, openings, "helpers")?,
            aux: next_field(&mut sequence, openings, "aux")?,
        })
    }
}

#[derive(Clone, Copy)]
struct RelationMessageSeed;

impl<'de> DeserializeSeed<'de> for RelationMessageSeed {
    type Value = RelationReceiptMessage;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "RelationReceiptMessage",
            &["opcode", "config", "roots", "external_inputs", "proof"],
            RelationMessageVisitor,
        )
    }
}

struct RelationMessageVisitor;

impl<'de> Visitor<'de> for RelationMessageVisitor {
    type Value = RelationReceiptMessage;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a relation receipt message")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(RelationReceiptMessage {
            opcode: next_field(&mut sequence, ValueSeed::<u16>::new(), "opcode")?,
            config: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<i64>::new()),
                "config",
            )?,
            roots: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<(Digest, Digest, u64)>::new()),
                "roots",
            )?,
            external_inputs: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<usize>::new()),
                "external_inputs",
            )?,
            proof: next_field(&mut sequence, RelationProofSeed, "proof")?,
        })
    }
}

#[derive(Clone, Copy)]
struct SharedTraceStepSeed;

impl<'de> DeserializeSeed<'de> for SharedTraceStepSeed {
    type Value = SharedTraceStep;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "SharedTraceStep",
            &["message", "external", "sources"],
            SharedTraceStepVisitor,
        )
    }
}

struct SharedTraceStepVisitor;

impl<'de> Visitor<'de> for SharedTraceStepVisitor {
    type Value = SharedTraceStep;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a shared trace step")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(SharedTraceStep {
            message: next_field(&mut sequence, RelationMessageSeed, "message")?,
            external: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<Option<usize>>::new()),
                "external",
            )?,
            sources: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<Option<(usize, usize)>>::new()),
                "sources",
            )?,
        })
    }
}

#[derive(Clone, Copy)]
struct SharedTraceSeed;

impl<'de> DeserializeSeed<'de> for SharedTraceSeed {
    type Value = SharedTrace;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "SharedTrace",
            &["tokens", "assets", "steps", "argmax", "output"],
            SharedTraceVisitor,
        )
    }
}

struct SharedTraceVisitor;

impl<'de> Visitor<'de> for SharedTraceVisitor {
    type Value = SharedTrace;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a shared trace")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        Ok(SharedTrace {
            tokens: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<u32>::new()),
                "tokens",
            )?,
            assets: next_field(&mut sequence, VecSeed::new(AssetOpeningSeed), "assets")?,
            steps: next_field(&mut sequence, VecSeed::new(SharedTraceStepSeed), "steps")?,
            argmax: next_field(
                &mut sequence,
                VecSeed::new(ValueSeed::<u32>::new()),
                "argmax",
            )?,
            output: sequence
                .next_element_seed(VecSeed::new(ValueSeed::<i64>::new()))?
                .unwrap_or_default(),
        })
    }
}

fn options(limit: u64) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(limit)
        .with_little_endian()
}

fn decode_error(error: bincode::Error, exceeded: bool) -> Error {
    if exceeded || matches!(*error, bincode::ErrorKind::SizeLimit) {
        Error::ResourceLimit
    } else {
        Error::InvalidEncoding
    }
}

fn require_eof<R: Read>(reader: &mut R) -> Result<()> {
    let mut trailing = [0u8; 1];
    match reader.read(&mut trailing) {
        Ok(0) => Ok(()),
        Ok(_) => Err(Error::InvalidProof),
        Err(_) => Err(Error::Io),
    }
}

pub(crate) fn read_file(path: &Path, memory_budget_bytes: u64) -> Result<SharedTrace> {
    if memory_budget_bytes == 0 {
        return Err(Error::ResourceLimit);
    }
    let mut file = fs::File::open(path).map_err(|_| Error::IncompleteArtifact)?;
    let metadata = file.metadata().map_err(|_| Error::IncompleteArtifact)?;
    if !metadata.is_file() {
        return Err(Error::IncompleteArtifact);
    }
    if metadata.len() > memory_budget_bytes {
        return Err(Error::ResourceLimit);
    }
    // Decode through a bounded read buffer: the wire format is read as u64
    // words one at a time, so an unbuffered file turns every word into a small
    // read. The buffer is a fixed 64 KiB window; the full proof is never
    // buffered and the budget checks above and in the visitors are unchanged.
    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
    read_from(&mut reader, memory_budget_bytes)
}

pub(crate) fn read_from<R: Read>(reader: &mut R, memory_budget_bytes: u64) -> Result<SharedTrace> {
    read_from_measured(reader, memory_budget_bytes).map(|(trace, _)| trace)
}

fn read_from_measured<R: Read>(
    reader: &mut R,
    memory_budget_bytes: u64,
) -> Result<(SharedTrace, AllocationUsage)> {
    if memory_budget_bytes == 0 {
        return Err(Error::ResourceLimit);
    }
    let heap_limit = usize::try_from(memory_budget_bytes).unwrap_or(usize::MAX);
    let guard = BudgetGuard::enter(heap_limit)?;
    let trace = options(memory_budget_bytes)
        .deserialize_from_seed(SharedTraceSeed, &mut *reader)
        .map_err(|error| decode_error(error, guard.state().exceeded))?;
    require_eof(reader)?;
    Ok((trace, guard.usage()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use zkie_core::oracle::OracleId;
    use zkie_engine::pcs::request::link::LinkProof;
    use zkie_ops::proof::relations::{OpRelationProof, RelationLengths, RelationReceiptMessage};

    fn empty_trace() -> SharedTrace {
        SharedTrace {
            tokens: Vec::new(),
            assets: Vec::new(),
            steps: Vec::new(),
            argmax: Vec::new(),
            output: Vec::new(),
        }
    }

    fn nested_step() -> SharedTraceStep {
        SharedTraceStep {
            message: RelationReceiptMessage {
                opcode: 1,
                config: vec![2],
                roots: vec![(Digest([9; 32]), Digest([10; 32]), 1)],
                external_inputs: vec![0],
                proof: OpRelationProof {
                    lengths: RelationLengths {
                        inputs: vec![1],
                        outputs: Vec::new(),
                        raw: Vec::new(),
                        quotients: Vec::new(),
                        helpers: Vec::new(),
                        aux: Vec::new(),
                    },
                    inputs: vec![vec![OracleOpening {
                        oracle: OracleId(3),
                        index: 0,
                        values: vec![11],
                        path: vec![Digest([12; 32])],
                    }]],
                    outputs: Vec::new(),
                    raw: Vec::new(),
                    quotients: Vec::new(),
                    helpers: Vec::new(),
                    aux: Vec::new(),
                },
            },
            external: vec![Some(0)],
            sources: vec![None],
        }
    }

    fn nested_trace() -> SharedTrace {
        SharedTrace {
            tokens: vec![1, 2],
            assets: vec![AssetOpening {
                asset_id: 7,
                link: LinkProof {
                    anchor_id: Digest([1; 32]),
                    registry_index: 0,
                    record: LeafRecord {
                        leaf_id: 7,
                        generation: 1,
                        table_identity: Digest([2; 32]),
                        leaf_root: Digest([3; 32]),
                    },
                    binding: Digest([4; 32]),
                    path: vec![Digest([5; 32])],
                },
                rows: vec![RowOpening {
                    row: 0,
                    values: vec![9, 10],
                    path: vec![Digest([6; 32])],
                }],
                start: Digest([7; 32]),
                root: Digest([8; 32]),
                row_count: 1,
            }],
            steps: vec![nested_step()],
            argmax: vec![0],
            output: vec![9],
        }
    }

    fn cumulative_trace() -> SharedTrace {
        let mut trace = nested_trace();
        trace.steps[0].message.proof.inputs = vec![Vec::new(); 64];
        trace
    }

    #[test]
    fn valid_empty_and_nested_traces_round_trip() {
        for expected in [empty_trace(), nested_trace()] {
            let bytes = bincode::serialize(&expected).expect("trace encoding");
            let mut reader = Cursor::new(bytes);
            let actual = read_from(&mut reader, 1 << 20).expect("bounded trace");
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn encoded_size_limit_is_enforced() {
        let bytes = bincode::serialize(&nested_trace()).expect("trace encoding");
        let limit = u64::try_from(bytes.len() - 1).expect("length");
        let mut reader = Cursor::new(bytes);
        assert_eq!(read_from(&mut reader, limit), Err(Error::ResourceLimit));
    }

    #[test]
    fn file_size_limit_is_enforced_before_decode() {
        let path = std::env::temp_dir().join(format!(
            "zkie-bounded-trace-size-{}.bin",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        fs::write(&path, [0u8; 16]).expect("proof fixture");
        assert_eq!(read_file(&path, 15), Err(Error::ResourceLimit));
        fs::remove_file(path).expect("fixture cleanup");
    }

    #[test]
    fn allocation_usage_records_cumulative_and_per_collection_bytes() {
        let bytes = bincode::serialize(&nested_trace()).expect("trace encoding");
        let (_, usage) = read_from_measured(&mut Cursor::new(bytes), 1 << 20)
            .expect("bounded trace with measurements");
        assert!(usage.allocations > 1);
        assert!(usage.charged_bytes > usage.largest_allocation_bytes);
        assert!(usage.largest_allocation_bytes > 0);
        assert!(usage.charged_bytes <= 1 << 20);
        eprintln!(
            "allocation_usage charged={} largest={} allocations={}",
            usage.charged_bytes, usage.largest_allocation_bytes, usage.allocations
        );
    }

    #[test]
    fn cumulative_limit_rejects_individually_small_collections() {
        let bytes = bincode::serialize(&cumulative_trace()).expect("trace encoding");
        let (_, usage) = read_from_measured(&mut Cursor::new(bytes.clone()), 1 << 20)
            .expect("measurement decode");
        let limit = bytes.len().max(usage.largest_allocation_bytes);
        assert!(limit < usage.charged_bytes);
        assert_eq!(
            read_from(&mut Cursor::new(bytes), limit as u64),
            Err(Error::ResourceLimit)
        );
    }

    #[test]
    fn top_level_collection_limit_is_checked_before_elements() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2_097_152u64.to_le_bytes());
        let mut reader = Cursor::new(bytes);
        assert_eq!(read_from(&mut reader, 65_536), Err(Error::ResourceLimit));
    }

    #[test]
    fn nested_collection_limit_is_checked_before_elements() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        bytes.extend_from_slice(&[0u8; 32]);
        bytes.extend_from_slice(&[0u8; 32]);
        bytes.extend_from_slice(&2_097_152u64.to_le_bytes());
        let mut reader = Cursor::new(bytes);
        assert_eq!(read_from(&mut reader, 65_536), Err(Error::ResourceLimit));
    }

    #[test]
    fn malformed_and_trailing_encodings_are_distinct() {
        let bytes = bincode::serialize(&nested_trace()).expect("trace encoding");
        let mut truncated = bytes.clone();
        truncated.pop();
        assert_eq!(
            read_from(&mut Cursor::new(truncated), 1 << 20),
            Err(Error::InvalidEncoding)
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            read_from(&mut Cursor::new(trailing), 1 << 20),
            Err(Error::InvalidProof)
        );
    }

    struct ReadOnlyChunks {
        bytes: Vec<u8>,
        position: usize,
        delivered: usize,
    }

    impl Read for ReadOnlyChunks {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.position == self.bytes.len() {
                return Ok(0);
            }
            let count = out.len().min(3).min(self.bytes.len() - self.position);
            out[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
            self.position += count;
            self.delivered += count;
            Ok(count)
        }
    }

    #[test]
    fn read_only_stream_is_decoded_without_rewind() {
        let expected = nested_trace();
        let bytes = bincode::serialize(&expected).expect("trace encoding");
        let encoded_len = bytes.len();
        let mut reader = ReadOnlyChunks {
            bytes,
            position: 0,
            delivered: 0,
        };
        let actual = read_from(&mut reader, 1 << 20).expect("single-pass decode");
        assert_eq!(actual, expected);
        assert_eq!(reader.position, encoded_len);
        assert_eq!(reader.delivered, encoded_len);
    }

    struct TailMutationReader {
        bytes: Vec<u8>,
        position: usize,
        mutated: bool,
    }

    impl Read for TailMutationReader {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.position == self.bytes.len() {
                return Ok(0);
            }
            let count = out.len().min(8).min(self.bytes.len() - self.position);
            out[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
            self.position += count;
            if self.position >= 8 && !self.mutated {
                self.bytes[8..16].copy_from_slice(&2_097_152u64.to_le_bytes());
                self.mutated = true;
            }
            Ok(count)
        }
    }

    #[test]
    fn in_flight_mutation_is_checked_in_the_same_forward_pass() {
        let bytes = bincode::serialize(&empty_trace()).expect("trace encoding");
        let mut reader = TailMutationReader {
            bytes,
            position: 0,
            mutated: false,
        };
        assert_eq!(
            read_from(&mut reader, 65_536),
            Err(Error::ResourceLimit)
        );
        assert!(reader.mutated);
    }

    #[test]
    fn malformed_length_error_is_deterministic() {
        let bytes = 2_097_152u64.to_le_bytes();
        for _ in 0..3 {
            assert_eq!(
                read_from(&mut Cursor::new(bytes), 65_536),
                Err(Error::ResourceLimit)
            );
        }
    }
}
