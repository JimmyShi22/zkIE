//! The authoritative cross-request commitment and opening ledger.
//!
//! The ledger is a durable record with two cooperating layers that share one
//! state file:
//!
//! - the commitment layer records an actual commitment root once, the table
//!   reuse policy and the per-request attempt state;
//! - the opening layer records every opening event by real oracle identity,
//!   the random material stream that produced the oracle, the logical draw
//!   count, the proof response entries and the distinct physical positions,
//!   and the exact union of opened positions per committed tree.
//!
//! The opening-layer counts are kept separate on purpose. A logical draw is one
//! transcript position draw and a repeated draw of the same column is counted
//! again. A response entry is one position carried by the proof message; the
//! runtime trees deduplicate their position lists while the star rows keep the
//! raw list. The unique physical positions of one tree are the union of every
//! stage's openings. For a product-form codeword the hiding budget is the code
//! tail rank `r`: one codeword symbol is one linear equation in the `r` random
//! tail coefficients plus the message coefficients. The security-relevant
//! check is therefore the union of distinct positions against `r`, while the
//! cost accounting charges every logical draw.
//!
//! The generation stream of one oracle is identified by a material commitment,
//! never by the observation that two draws produced equal positions or equal
//! field values. Equal positions and equal values are accepted; a repeated
//! material commitment is not. The request carries a globally unique seed
//! commitment, which is reserved before any proof work. Per-oracle material
//! commitments are derived from it by the caller and are also one-use.
//!
//! Updates use an exclusive lock, a temporary file, a file sync, a rename and
//! a parent-directory sync. A crash before the rename leaves the old state
//! plus a lock or a temporary file, and every later reservation fails closed.
//! A crash after the rename leaves the spent state or a lock that also fails
//! closed. There is no rollback API. A filesystem rollback to an earlier
//! directory snapshot is an external deployment trust assumption and is not
//! detected here.
//!
//! Lifecycle order is strict: a request is `Reserved`, becomes `Proved` after
//! the prover returns, `Verified` only after the complete verification accepts,
//! and `Succeeded` only after `Verified`. A failed or crashed reservation stays
//! spent forever. The event log is append-only with a sequence number, the
//! previous event digest and a state head digest; a truncated log, a sequence
//! regression or an edited event is rejected when the state is opened.
//!
//! The caller sequence around one proof is therefore fixed: begin the request
//! and reserve every oracle material before any proof work; bind each actual
//! root when the prover produces it; record each opening event when its drawn
//! positions are known; mark the request proved when the prover returns; and
//! mark it verified and succeeded only after the complete verifier accepts.
//! The hidden output-head prover and verifier are pure cryptographic entries
//! and do not receive the ledger; the callable formal fixture drives this
//! sequence around them, and a deployment entry must do the same.
//!
//! Accounting of the hidden output-head path, one formal run of 16 positions.
//! The pre-closure protocol (before the exact integer output closure) had the
//! following logical draws and distinct positions:
//!
//! ```text
//! category                    logical  unique
//! activation rows                   0       0   (16 rows allocated, never opened)
//! table columns                   603     601 *
//! W SEND (16)                      32      32
//! W BASE (16)                      32      32
//! index mask SEND                   2       2
//! index mask BASE                   2       2
//! shown map BASE                    2       2
//! Omega shared BASE                 2       2
//! Omega own BASE                    2       2
//! G_u shared BASE                   2       2
//! fresh mask shared BASE            2       2
//! fresh mask own BASE               2       2
//! star + star mask shared           4       4
//! star + star mask own              4       4
//! switch / CS                       0       0
//! total                           691     689 *
//! ```
//!
//! `*` The table challenge is independent of the closure. The reference
//! pre-closure artifact repeated two table columns (603 draws, 601 distinct);
//! the closed formal run measured in this repository drew 603 distinct columns
//! of 603 draws and therefore had 691 distinct positions. Both measurements
//! are recorded; neither replaces the other.
//!
//! The closed output relation adds three committed value messages (slack,
//! packed bits, product map), each with 16 oracles opened at SEND and at own
//! BASE, plus the mask-only closure message opened at SEND and at own BASE.
//! With two SEND openings, two own-BASE positions and two closure openings per
//! oracle the added accounting is:
//!
//! ```text
//! category                    logical  unique
//! closure slack SEND (16)          32      32
//! closure slack BASE (16)          32      32
//! closure bits SEND (16)           32      32
//! closure bits BASE (16)           32      32
//! closure map SEND (16)            32      32
//! closure map BASE (16)            32      32
//! closure mask SEND                 2       2
//! closure mask BASE                 2       2
//! added                           196     196
//! closed total                    887     885 *
//! ```
//!
//! The previous ledger charged 655 logical openings: 16 activation rows that
//! are never opened, 603 logical table draws, 32 W SEND draws, 2 index-mask
//! SEND draws and 2 shown-map BASE draws. The audited opening count of 673 is
//! 601 distinct table columns plus 32 W SEND plus 2 index-mask SEND plus 38
//! own-group BASE openings for the artifact whose table draw repeated two
//! columns; over the closed run's 603 distinct columns the same decomposition
//! is 675. The 36-versus-70 and 655-versus-673 differences are therefore not
//! counting errors of the proof: the old ledger mixed allocations, logical
//! draws and unique positions and had no key for the own-group BASE member
//! openings. This layer records the three counts separately and the exact
//! union per tree, so no category is conflated.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::lm_head_stage::{
    decode_bytes, encode_bytes, ensure_private_dir, private_options, read_bytes, sync_dir,
    write_bytes,
};

/// The ledger schema version. It covers the state file and every record.
pub const COMMITMENT_LEDGER_VERSION: u64 = 3;

/// The opening-event schema version.
pub const LEDGER_EVENT_SCHEMA_VERSION: u64 = 1;

const LEDGER_FILE: &str = "commitments.state";
const LEDGER_LOCK: &str = "commitments.lock";
const LEDGER_TEMP: &str = "commitments.tmp";

/// How long a reservation waits for a concurrent writer before failing closed.
const LOCK_ATTEMPTS: u32 = 100;
const LOCK_RETRY: Duration = Duration::from_millis(2);

/// The kind of commitment a commitment-layer entry names. A non-table
/// commitment may be reserved once and is then spent; a table commitment may
/// be reused only when the ledger policy explicitly budgets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CommitmentKind {
    /// The hidden activation commitment `X`.
    Activation,
    /// One hidden reduced-weight commitment `W`.
    Weight,
    /// The committed encoded table.
    Table,
    /// A hiding mask commitment.
    Mask,
    /// The committed shown-map message of the hidden Dot sumcheck.
    Dot,
}

impl CommitmentKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Activation => "activation",
            Self::Weight => "weight",
            Self::Table => "table",
            Self::Mask => "mask",
            Self::Dot => "dot",
        }
    }
}

/// The kind of oracle message whose opening events the ledger charges. The
/// two switch and CS kinds are reserved for protocols that carry those
/// messages; the hidden output-head path has zero events of either kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OracleKind {
    Activation,
    Weight,
    Table,
    IndexMask,
    DotMap,
    Omega,
    Gu,
    FreshMask,
    Star,
    StarMask,
    Fold,
    Cs,
    Switch,
    ClosureSlack,
    ClosureBits,
    ClosureMap,
    ClosureMask,
}

impl OracleKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Activation => "activation",
            Self::Weight => "weight",
            Self::Table => "table",
            Self::IndexMask => "index-mask",
            Self::DotMap => "dot-map",
            Self::Omega => "omega",
            Self::Gu => "gu",
            Self::FreshMask => "fresh-mask",
            Self::Star => "star",
            Self::StarMask => "star-mask",
            Self::Fold => "fold",
            Self::Cs => "cs",
            Self::Switch => "switch",
            Self::ClosureSlack => "closure-slack",
            Self::ClosureBits => "closure-bits",
            Self::ClosureMap => "closure-map",
            Self::ClosureMask => "closure-mask",
        }
    }

    pub const fn code(self) -> u64 {
        match self {
            Self::Activation => 1,
            Self::Weight => 2,
            Self::Table => 3,
            Self::IndexMask => 4,
            Self::DotMap => 5,
            Self::Omega => 6,
            Self::Gu => 7,
            Self::FreshMask => 8,
            Self::Star => 9,
            Self::StarMask => 10,
            Self::Fold => 11,
            Self::Cs => 12,
            Self::Switch => 13,
            Self::ClosureSlack => 14,
            Self::ClosureBits => 15,
            Self::ClosureMap => 16,
            Self::ClosureMask => 17,
        }
    }
}

/// The stage that produced one opening event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stage {
    Send,
    Shared,
    Second,
    Switch,
    Cs,
    Base,
    Table,
}

impl Stage {
    pub const fn code(self) -> u64 {
        match self {
            Self::Send => 1,
            Self::Shared => 2,
            Self::Second => 3,
            Self::Switch => 4,
            Self::Cs => 5,
            Self::Base => 6,
            Self::Table => 7,
        }
    }
}

/// The transcript source of one opening event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OpeningSource {
    SendSample,
    BaseMember,
    BaseFresh,
    BaseStar,
    TableColumn,
    Switch,
    Cs,
}

impl OpeningSource {
    pub const fn code(self) -> u64 {
        match self {
            Self::SendSample => 1,
            Self::BaseMember => 2,
            Self::BaseFresh => 3,
            Self::BaseStar => 4,
            Self::TableColumn => 5,
            Self::Switch => 6,
            Self::Cs => 7,
        }
    }
}

/// The key of one commitment entry: its kind and its root or identity. The
/// request or attempt is recorded inside the entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LedgerKey {
    pub kind: CommitmentKind,
    pub identity: Digest,
}

impl LedgerKey {
    pub fn new(kind: CommitmentKind, identity: Digest) -> Result<Self> {
        if identity == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { kind, identity })
    }
}

/// The table-reuse policy fixed when the ledger is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableReusePolicy {
    /// A table root may serve exactly one request.
    Forbidden,
    /// A table root may serve up to `max_requests` requests and up to
    /// `max_logical_openings` logical column openings in total.
    Budget {
        max_requests: u64,
        max_logical_openings: u64,
    },
}

/// The ledger policy. It is stored in the state file and cannot change after
/// initialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerPolicy {
    pub table_reuse: TableReusePolicy,
}

impl Default for LedgerPolicy {
    fn default() -> Self {
        Self {
            table_reuse: TableReusePolicy::Forbidden,
        }
    }
}

impl LedgerPolicy {
    fn validate(&self) -> Result<()> {
        match self.table_reuse {
            TableReusePolicy::Forbidden => Ok(()),
            TableReusePolicy::Budget {
                max_requests,
                max_logical_openings,
            } => {
                if max_requests == 0 || max_logical_openings == 0 {
                    return Err(Error::InvalidConfig);
                }
                Ok(())
            }
        }
    }
}

/// The state of one commitment-layer request or attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptState {
    /// The reservation is durable and the proof work has not finished.
    Reserved,
    /// The proof work finished and the caller recorded success.
    Succeeded,
    /// The proof work finished and the caller recorded failure.
    Failed,
}

/// The lifecycle state of one request in the opening layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    /// The seed commitment and the oracle streams are durable; proof work has
    /// not finished.
    Reserved,
    /// The prover returned, but the verification has not accepted yet.
    Proved,
    /// The complete verification accepted.
    Verified,
    /// The request is closed and can never be retried.
    Succeeded,
    /// The request is closed and can never be retried.
    Failed,
}

impl RequestState {
    pub const fn code(self) -> u64 {
        match self {
            Self::Reserved => 1,
            Self::Proved => 2,
            Self::Verified => 3,
            Self::Succeeded => 4,
            Self::Failed => 5,
        }
    }
}

/// The lifecycle state of one oracle in the opening layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventState {
    /// The material stream is reserved and no opening was recorded.
    Reserved,
    /// At least one opening event was recorded.
    Opened,
    /// The actual commitment root is bound.
    Bound,
    /// The prover returned.
    Proved,
    /// The complete verification accepted.
    Verified,
    /// The request succeeded.
    Succeeded,
    /// The request or the opening failed; the material stays spent.
    Failed,
    /// A tail-union overrun aborted the oracle; the material stays spent.
    Aborted,
}

impl EventState {
    pub const fn code(self) -> u64 {
        match self {
            Self::Reserved => 1,
            Self::Opened => 2,
            Self::Bound => 3,
            Self::Proved => 4,
            Self::Verified => 5,
            Self::Succeeded => 6,
            Self::Failed => 7,
            Self::Aborted => 8,
        }
    }
}

/// The outcome passed to [`CommitmentLedger::finish_request`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestOutcome {
    /// The prover returned successfully.
    Proved,
    /// The complete verifier accepted.
    Verified,
    /// The request closed successfully; it must follow `Verified`.
    Succeeded,
    /// The request failed; the material stays spent.
    Failed,
}

/// The key of one oracle message. `identity` is the root when the root exists
/// before proof work, otherwise a stable plan identity; the actual root is
/// bound later by [`CommitmentLedger::bind_commitment`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OracleKey {
    pub kind: OracleKind,
    pub identity: Digest,
    pub oracle_id: u64,
    pub tree_number: u64,
    pub run: Digest,
    pub configuration: Digest,
}

impl OracleKey {
    pub fn new(
        kind: OracleKind,
        identity: Digest,
        oracle_id: u64,
        tree_number: u64,
        run: Digest,
        configuration: Digest,
    ) -> Result<Self> {
        if identity == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            kind,
            identity,
            oracle_id,
            tree_number,
            run,
            configuration,
        })
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        put_u64(out, self.kind.code());
        put_digest(out, &self.identity);
        put_u64(out, self.oracle_id);
        put_u64(out, self.tree_number);
        put_digest(out, &self.run);
        put_digest(out, &self.configuration);
    }
}

/// The identity of the random material that produced one oracle message. The
/// material commitment is derived from the request seed by the caller and is
/// never the seed itself. Two messages with different roots must use different
/// material commitments; the same material commitment is one-use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StreamKey {
    pub kind: OracleKind,
    pub oracle_id: u64,
    pub tree_number: u64,
    pub run: Digest,
    pub configuration: Digest,
    pub material: Digest,
}

impl StreamKey {
    pub fn new(
        kind: OracleKind,
        oracle_id: u64,
        tree_number: u64,
        run: Digest,
        configuration: Digest,
        material: Digest,
    ) -> Result<Self> {
        if material == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            kind,
            oracle_id,
            tree_number,
            run,
            configuration,
            material,
        })
    }

    fn matches(&self, oracle: &OracleKey) -> bool {
        self.kind == oracle.kind
            && self.oracle_id == oracle.oracle_id
            && self.tree_number == oracle.tree_number
            && self.run == oracle.run
            && self.configuration == oracle.configuration
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        put_u64(out, self.kind.code());
        put_u64(out, self.oracle_id);
        put_u64(out, self.tree_number);
        put_digest(out, &self.run);
        put_digest(out, &self.configuration);
        put_digest(out, &self.material);
    }
}

/// The three opening counts of one event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpeningCounts {
    /// Transcript position draws; a repeated draw of the same column counts
    /// again.
    pub logical_draws: u64,
    /// Positions carried by the proof message. Runtime trees deduplicate their
    /// lists; the star rows keep the raw drawn list.
    pub response_entries: u64,
    /// Distinct physical positions opened by this event.
    pub unique_positions: u64,
}

impl OpeningCounts {
    pub fn new(logical_draws: u64, response_entries: u64, unique_positions: u64) -> Result<Self> {
        if response_entries < unique_positions
            || logical_draws < response_entries
            || unique_positions > logical_draws
        {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            logical_draws,
            response_entries,
            unique_positions,
        })
    }

    fn add_into(&mut self, other: OpeningCounts) -> Result<()> {
        self.logical_draws = self
            .logical_draws
            .checked_add(other.logical_draws)
            .ok_or(Error::ResourceLimit)?;
        self.response_entries = self
            .response_entries
            .checked_add(other.response_entries)
            .ok_or(Error::ResourceLimit)?;
        self.unique_positions = self
            .unique_positions
            .checked_add(other.unique_positions)
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }
}

/// The tail-union state of one event. `rank` is the code tail rank `r`, or
/// zero for a message without a code tail (activation rows and fresh star
/// rows). `remaining` is `rank - new_union` for a code message and zero
/// otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TailState {
    pub rank: u64,
    pub prior_union: u64,
    pub new_union: u64,
    pub remaining: u64,
}

/// The declared budget of one oracle reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleBudget {
    /// The maximum logical draws of the oracle across every stage.
    pub logical_draws: u64,
    /// The code tail rank; zero for a message without a code tail.
    pub tail_rank: u64,
    /// Rows allocated by the commitment but never opened in this path.
    pub allocated_rows: u64,
    /// One-time committed message entries (mask entries plus pads).
    pub committed_entries: u64,
}

impl OracleBudget {
    pub fn new(logical_draws: u64, tail_rank: u64) -> Result<Self> {
        if logical_draws == 0 {
            return Err(Error::InvalidConfig);
        }
        if tail_rank != 0 && logical_draws > tail_rank {
            return Err(Error::LeafBudgetExhausted);
        }
        Ok(Self {
            logical_draws,
            tail_rank,
            allocated_rows: 0,
            committed_entries: 0,
        })
    }

    pub fn with_allocation(mut self, allocated_rows: u64, committed_entries: u64) -> Self {
        self.allocated_rows = allocated_rows;
        self.committed_entries = committed_entries;
        self
    }
}

/// One committed opening event. The event is appended in sequence; the event
/// id chains the previous id, the request, the oracle, the counts, the tail
/// state and the exact distinct positions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub schema: u64,
    pub seq: u64,
    pub prev_event: Digest,
    pub event_id: Digest,
    pub request: Digest,
    pub attempt: u64,
    pub oracle: OracleKey,
    pub stream: StreamKey,
    pub stage: Stage,
    pub source: OpeningSource,
    pub counts: OpeningCounts,
    pub tail: TailState,
    pub positions: Vec<u64>,
    pub state: EventState,
}

impl LedgerEvent {
    fn compute_id(&self) -> Digest {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(b"zkie/commitment-ledger/event");
        put_u64(&mut out, self.schema);
        put_u64(&mut out, self.seq);
        put_digest(&mut out, &self.prev_event);
        put_digest(&mut out, &self.request);
        put_u64(&mut out, self.attempt);
        self.oracle.encode_into(&mut out);
        self.stream.encode_into(&mut out);
        put_u64(&mut out, self.stage.code());
        put_u64(&mut out, self.source.code());
        put_u64(&mut out, self.counts.logical_draws);
        put_u64(&mut out, self.counts.response_entries);
        put_u64(&mut out, self.counts.unique_positions);
        put_u64(&mut out, self.tail.rank);
        put_u64(&mut out, self.tail.prior_union);
        put_u64(&mut out, self.tail.new_union);
        put_u64(&mut out, self.tail.remaining);
        put_u64(&mut out, self.positions.len() as u64);
        for position in &self.positions {
            put_u64(&mut out, *position);
        }
        put_u64(&mut out, self.state.code());
        digest_bytes(&out)
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_digest(out: &mut Vec<u8>, value: &Digest) {
    out.extend_from_slice(&value.0);
}

/// One commitment-layer request or attempt record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub state: AttemptState,
    pub logical_openings: u64,
}

/// One commitment entry. `commitment` is the actual commitment recorded once;
/// `identity` in the key is its root or a stable identity derived from it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentRecord {
    pub commitment: Digest,
    pub logical_openings: u64,
    pub table_reuse: u64,
    pub attempts: BTreeMap<Digest, AttemptRecord>,
}

impl CommitmentRecord {
    pub fn attempt(&self, request: Digest) -> Option<&AttemptRecord> {
        self.attempts.get(&request)
    }

    /// The number of reservations after the first one. It is always zero for
    /// a non-table commitment.
    pub fn table_reuse_count(&self) -> u64 {
        self.table_reuse
    }
}

/// One oracle record of the opening layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleRecord {
    pub commitment: Option<Digest>,
    pub stream: StreamKey,
    pub tail_rank: u64,
    pub allocated_rows: u64,
    pub committed_entries: u64,
    pub opened_positions: BTreeSet<u64>,
    pub logical_openings: u64,
    pub response_entries: u64,
    pub unique_positions: u64,
    pub declared_logical: u64,
    pub first_request: Digest,
    pub requests: BTreeSet<Digest>,
    pub per_stage: BTreeMap<Stage, OpeningCounts>,
    pub state: EventState,
}

impl OracleRecord {
    /// The remaining hiding budget of this tree: `rank - union`, or zero for a
    /// message without a code tail.
    pub fn remaining_tail(&self) -> u64 {
        if self.tail_rank == 0 {
            0
        } else {
            self.tail_rank.saturating_sub(self.unique_positions)
        }
    }

    pub fn totals(&self) -> OpeningCounts {
        OpeningCounts {
            logical_draws: self.logical_openings,
            response_entries: self.response_entries,
            unique_positions: self.unique_positions,
        }
    }
}

/// One random material stream record. The material is one-use: a non-table
/// stream serves exactly one request, and a table stream serves only the
/// budgeted reuse of its own table oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamRecord {
    pub oracle: OracleKey,
    pub first_request: Digest,
}

/// One request record of the opening layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestRecord {
    pub state: RequestState,
    pub seed_commitment: Digest,
    pub plan_digest: Digest,
    pub oracles: BTreeSet<OracleKey>,
    pub events: Vec<u64>,
}

impl RequestRecord {
    /// The number of opening events recorded for this request.
    pub fn event_count(&self) -> u64 {
        self.events.len() as u64
    }
}

/// The complete durable ledger state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerState {
    pub version: u64,
    pub policy: LedgerPolicy,
    /// The sequence number of the last event; zero when the log is empty.
    pub seq: u64,
    /// The digest of the last event; zero when the log is empty.
    pub head: Digest,
    pub commitments: BTreeMap<LedgerKey, CommitmentRecord>,
    pub oracles: BTreeMap<OracleKey, OracleRecord>,
    pub streams: BTreeMap<StreamKey, StreamRecord>,
    pub requests: BTreeMap<Digest, RequestRecord>,
    pub events: Vec<LedgerEvent>,
}

/// The tail-hiding budget over every code-bearing oracle. `min_remaining` is
/// the smallest `rank - union`; zero means at least one oracle has no unused
/// tail coordinate. Equality is a planning condition, not an automatic
/// failure: the caller must include every opening in the union and argue the
/// equality (the square Vandermonde case), or fail closed. `record_openings`
/// rejects any union above the rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpeningBudgetSummary {
    pub code_oracles: u64,
    pub min_remaining: u64,
    pub total_rank: u64,
    pub total_unique: u64,
    pub exhausted: u64,
}

impl LedgerState {
    pub fn record(&self, key: LedgerKey) -> Option<&CommitmentRecord> {
        self.commitments.get(&key)
    }

    pub fn oracle(&self, key: OracleKey) -> Option<&OracleRecord> {
        self.oracles.get(&key)
    }

    pub fn request(&self, request: Digest) -> Option<&RequestRecord> {
        self.requests.get(&request)
    }

    /// The total logical openings recorded by every opening event.
    pub fn total_logical_openings(&self) -> Result<u64> {
        let mut total = 0u64;
        for record in self.commitments.values() {
            total = total
                .checked_add(record.logical_openings)
                .ok_or(Error::ResourceLimit)?;
        }
        for record in self.oracles.values() {
            total = total
                .checked_add(record.logical_openings)
                .ok_or(Error::ResourceLimit)?;
        }
        Ok(total)
    }

    /// The total logical draws of the opening layer.
    pub fn total_event_draws(&self) -> Result<u64> {
        self.oracles.values().try_fold(0u64, |total, record| {
            total
                .checked_add(record.logical_openings)
                .ok_or(Error::ResourceLimit)
        })
    }

    /// The total distinct positions opened per tree, summed over trees. This
    /// is not a global deduplication: two trees may open the same column.
    pub fn total_event_unique_positions(&self) -> Result<u64> {
        self.oracles.values().try_fold(0u64, |total, record| {
            total
                .checked_add(record.unique_positions)
                .ok_or(Error::ResourceLimit)
        })
    }

    /// The total number of table reservations after the first one.
    pub fn total_table_reuse(&self) -> Result<u64> {
        self.commitments.values().try_fold(0u64, |total, record| {
            total
                .checked_add(record.table_reuse)
                .ok_or(Error::ResourceLimit)
        })
    }

    /// The tail-hiding budget over every code-bearing oracle.
    pub fn opening_budget_summary(&self) -> OpeningBudgetSummary {
        let mut summary = OpeningBudgetSummary {
            min_remaining: u64::MAX,
            ..OpeningBudgetSummary::default()
        };
        for record in self.oracles.values() {
            if record.tail_rank == 0 {
                continue;
            }
            let remaining = record.remaining_tail();
            summary.code_oracles = summary.code_oracles.saturating_add(1);
            summary.total_rank = summary.total_rank.saturating_add(record.tail_rank);
            summary.total_unique = summary.total_unique.saturating_add(record.unique_positions);
            summary.min_remaining = summary.min_remaining.min(remaining);
            if remaining == 0 {
                summary.exhausted = summary.exhausted.saturating_add(1);
            }
        }
        if summary.code_oracles == 0 {
            summary.min_remaining = 0;
        }
        summary
    }

    /// The opening totals of one oracle kind.
    pub fn totals_by_kind(&self, kind: OracleKind) -> Result<OpeningCounts> {
        let mut totals = OpeningCounts::default();
        for (key, record) in &self.oracles {
            if key.kind == kind {
                totals.add_into(record.totals())?;
            }
        }
        Ok(totals)
    }

    /// The opening totals of one stage.
    pub fn totals_by_stage(&self, stage: Stage) -> Result<OpeningCounts> {
        let mut totals = OpeningCounts::default();
        for record in self.oracles.values() {
            if let Some(counts) = record.per_stage.get(&stage) {
                totals.add_into(*counts)?;
            }
        }
        Ok(totals)
    }

    fn validate(&self) -> Result<()> {
        if self.version != COMMITMENT_LEDGER_VERSION {
            return Err(Error::UnsupportedVersion);
        }
        self.policy.validate()?;
        for (key, record) in &self.commitments {
            if key.identity == Digest::ZERO || record.commitment == Digest::ZERO {
                return Err(Error::InvalidEncoding);
            }
            let mut openings = 0u64;
            for (request, attempt) in &record.attempts {
                if *request == Digest::ZERO || attempt.logical_openings == 0 {
                    return Err(Error::InvalidEncoding);
                }
                openings = openings
                    .checked_add(attempt.logical_openings)
                    .ok_or(Error::ResourceLimit)?;
            }
            if openings != record.logical_openings {
                return Err(Error::StateRollbackSuspected);
            }
            match key.kind {
                CommitmentKind::Table => {
                    let expected_reuse = record.attempts.len().saturating_sub(1) as u64;
                    if record.table_reuse != expected_reuse {
                        return Err(Error::StateRollbackSuspected);
                    }
                    match self.policy.table_reuse {
                        TableReusePolicy::Forbidden => {
                            if record.attempts.len() > 1 {
                                return Err(Error::StateRollbackSuspected);
                            }
                        }
                        TableReusePolicy::Budget {
                            max_requests,
                            max_logical_openings,
                        } => {
                            if record.attempts.len() as u64 > max_requests
                                || record.logical_openings > max_logical_openings
                            {
                                return Err(Error::StateRollbackSuspected);
                            }
                        }
                    }
                }
                CommitmentKind::Activation
                | CommitmentKind::Weight
                | CommitmentKind::Mask
                | CommitmentKind::Dot => {
                    if record.attempts.len() > 1 || record.table_reuse != 0 {
                        return Err(Error::StateRollbackSuspected);
                    }
                }
            }
        }
        let mut seeds = BTreeSet::new();
        for (request, record) in &self.requests {
            if *request == Digest::ZERO
                || record.seed_commitment == Digest::ZERO
                || record.plan_digest == Digest::ZERO
            {
                return Err(Error::InvalidEncoding);
            }
            if !seeds.insert(record.seed_commitment) {
                return Err(Error::StateRollbackSuspected);
            }
            let mut previous = 0u64;
            for seq in &record.events {
                if *seq == 0 || *seq <= previous || *seq > self.events.len() as u64 {
                    return Err(Error::StateRollbackSuspected);
                }
                previous = *seq;
            }
            for oracle in &record.oracles {
                let entry = self
                    .oracles
                    .get(oracle)
                    .ok_or(Error::StateRollbackSuspected)?;
                if !entry.requests.contains(request) {
                    return Err(Error::StateRollbackSuspected);
                }
            }
        }
        let mut bound_roots = BTreeSet::new();
        for (key, record) in &self.oracles {
            if key.identity == Digest::ZERO || !record.stream.matches(key) {
                return Err(Error::InvalidEncoding);
            }
            if record.requests.is_empty() || !record.requests.contains(&record.first_request) {
                return Err(Error::StateRollbackSuspected);
            }
            for request in &record.requests {
                let entry = self
                    .requests
                    .get(request)
                    .ok_or(Error::StateRollbackSuspected)?;
                if !entry.oracles.contains(key) {
                    return Err(Error::StateRollbackSuspected);
                }
            }
            if let Some(commitment) = record.commitment {
                if commitment == Digest::ZERO || !bound_roots.insert(commitment) {
                    return Err(Error::StateRollbackSuspected);
                }
            }
            if record.unique_positions != record.opened_positions.len() as u64 {
                return Err(Error::StateRollbackSuspected);
            }
            if record.tail_rank != 0 && record.unique_positions > record.tail_rank {
                return Err(Error::StateRollbackSuspected);
            }
            if record.response_entries < record.unique_positions
                || record.logical_openings < record.response_entries
            {
                return Err(Error::StateRollbackSuspected);
            }
            if record.logical_openings > record.declared_logical {
                return Err(Error::StateRollbackSuspected);
            }
            let mut stage_logical = 0u64;
            let mut stage_response = 0u64;
            for counts in record.per_stage.values() {
                stage_logical = stage_logical
                    .checked_add(counts.logical_draws)
                    .ok_or(Error::ResourceLimit)?;
                stage_response = stage_response
                    .checked_add(counts.response_entries)
                    .ok_or(Error::ResourceLimit)?;
            }
            if stage_logical != record.logical_openings || stage_response != record.response_entries
            {
                return Err(Error::StateRollbackSuspected);
            }
            if key.kind == OracleKind::Table {
                match self.policy.table_reuse {
                    TableReusePolicy::Forbidden => {
                        if record.requests.len() != 1 {
                            return Err(Error::StateRollbackSuspected);
                        }
                    }
                    TableReusePolicy::Budget {
                        max_requests,
                        max_logical_openings,
                    } => {
                        if record.requests.len() as u64 > max_requests
                            || record.declared_logical > max_logical_openings
                        {
                            return Err(Error::StateRollbackSuspected);
                        }
                    }
                }
            } else if record.requests.len() != 1 {
                return Err(Error::StateRollbackSuspected);
            }
        }
        for (stream, record) in &self.streams {
            if stream.material == Digest::ZERO || !stream.matches(&record.oracle) {
                return Err(Error::InvalidEncoding);
            }
            let oracle = self
                .oracles
                .get(&record.oracle)
                .ok_or(Error::StateRollbackSuspected)?;
            if oracle.stream != *stream {
                return Err(Error::StateRollbackSuspected);
            }
            let request = self
                .requests
                .get(&record.first_request)
                .ok_or(Error::StateRollbackSuspected)?;
            if !request.oracles.contains(&record.oracle) {
                return Err(Error::StateRollbackSuspected);
            }
        }
        self.validate_events()
    }

    fn validate_events(&self) -> Result<()> {
        if self.seq != self.events.len() as u64 {
            return Err(Error::StateRollbackSuspected);
        }
        let mut head = Digest::ZERO;
        let mut unions: BTreeMap<OracleKey, BTreeSet<u64>> = BTreeMap::new();
        let mut totals: BTreeMap<OracleKey, (OpeningCounts, BTreeMap<Stage, OpeningCounts>)> =
            BTreeMap::new();
        let mut request_seqs: BTreeMap<Digest, Vec<u64>> = BTreeMap::new();
        for (index, event) in self.events.iter().enumerate() {
            let seq = index as u64 + 1;
            if event.schema != LEDGER_EVENT_SCHEMA_VERSION
                || event.seq != seq
                || event.prev_event != head
                || event.event_id != event.compute_id()
            {
                return Err(Error::StateRollbackSuspected);
            }
            head = event.event_id;
            self.requests
                .get(&event.request)
                .ok_or(Error::StateRollbackSuspected)?;
            let oracle = self
                .oracles
                .get(&event.oracle)
                .ok_or(Error::StateRollbackSuspected)?;
            if oracle.stream != event.stream
                || !event.stream.matches(&event.oracle)
                || oracle.tail_rank != event.tail.rank
                || !oracle.requests.contains(&event.request)
            {
                return Err(Error::StateRollbackSuspected);
            }
            let sequences = request_seqs.entry(event.request).or_default();
            if event.attempt != sequences.len() as u64 + 1 {
                return Err(Error::StateRollbackSuspected);
            }
            sequences.push(event.seq);
            if event.counts.unique_positions != event.positions.len() as u64 {
                return Err(Error::StateRollbackSuspected);
            }
            let mut previous = None;
            for position in &event.positions {
                if let Some(last) = previous {
                    if *position <= last {
                        return Err(Error::StateRollbackSuspected);
                    }
                }
                previous = Some(*position);
            }
            if event.counts.response_entries < event.counts.unique_positions
                || event.counts.logical_draws < event.counts.response_entries
            {
                return Err(Error::StateRollbackSuspected);
            }
            let union = unions.entry(event.oracle).or_default();
            let prior_union = union.len() as u64;
            for position in &event.positions {
                union.insert(*position);
            }
            let new_union = union.len() as u64;
            let remaining = if event.tail.rank == 0 {
                0
            } else {
                event
                    .tail
                    .rank
                    .checked_sub(new_union)
                    .ok_or(Error::StateRollbackSuspected)?
            };
            if event.tail.prior_union != prior_union
                || event.tail.new_union != new_union
                || event.tail.remaining != remaining
            {
                return Err(Error::StateRollbackSuspected);
            }
            let entry = totals
                .entry(event.oracle)
                .or_insert_with(|| (OpeningCounts::default(), BTreeMap::new()));
            entry.0.add_into(event.counts)?;
            entry
                .1
                .entry(event.stage)
                .or_insert_with(OpeningCounts::default)
                .add_into(event.counts)?;
        }
        if self.head != head {
            return Err(Error::StateRollbackSuspected);
        }
        for (request, record) in &self.requests {
            let sequences = request_seqs.remove(request).unwrap_or_default();
            if sequences != record.events {
                return Err(Error::StateRollbackSuspected);
            }
        }
        if !request_seqs.is_empty() {
            return Err(Error::StateRollbackSuspected);
        }
        for (key, record) in &self.oracles {
            let union = unions.remove(key).unwrap_or_default();
            if union != record.opened_positions {
                return Err(Error::StateRollbackSuspected);
            }
            let (counts, stages) = totals
                .remove(key)
                .unwrap_or_else(|| (OpeningCounts::default(), BTreeMap::new()));
            if counts.logical_draws != record.logical_openings
                || counts.response_entries != record.response_entries
                || stages != record.per_stage
            {
                return Err(Error::StateRollbackSuspected);
            }
        }
        if !unions.is_empty() || !totals.is_empty() {
            return Err(Error::StateRollbackSuspected);
        }
        Ok(())
    }
}

/// One durable reservation of the commitment layer. The reservation is already
/// committed when this value is returned; dropping it leaves the attempt in
/// [`AttemptState::Reserved`] and does not release anything. Call
/// [`LedgerClaim::succeed`] or [`LedgerClaim::fail`] to record the outcome of
/// the proof work.
#[derive(Debug)]
#[must_use = "the reservation is already durable; dropping the claim does not release it"]
pub struct LedgerClaim {
    dir: PathBuf,
    key: LedgerKey,
    request: Digest,
    logical_openings: u64,
    table_reuse: u64,
    attempt_state: AttemptState,
}

impl LedgerClaim {
    pub fn key(&self) -> LedgerKey {
        self.key
    }

    pub fn request(&self) -> Digest {
        self.request
    }

    /// The number of logical openings this reservation charged.
    pub fn logical_openings(&self) -> u64 {
        self.logical_openings
    }

    /// The table reuse count of this reservation: zero for the first use of a
    /// table root, one for its first reuse, and so on. It is always zero for a
    /// non-table commitment.
    pub fn table_reuse(&self) -> u64 {
        self.table_reuse
    }

    pub fn attempt_state(&self) -> AttemptState {
        self.attempt_state
    }

    /// Records that the proof work succeeded. The reservation itself stays
    /// spent either way.
    pub fn succeed(mut self) -> Result<()> {
        self.finish(AttemptState::Succeeded)
    }

    /// Records that the proof work failed. The reservation itself stays spent
    /// either way; a later retry with the same commitment is rejected.
    pub fn fail(mut self) -> Result<()> {
        self.finish(AttemptState::Failed)
    }

    fn finish(&mut self, state: AttemptState) -> Result<()> {
        finish_attempt(&self.dir, self.key, self.request, state)?;
        self.attempt_state = state;
        Ok(())
    }
}

/// The durable ledger directory. Cloning the handle is cheap and every process
/// that opens the same directory shares the same authoritative state.
#[derive(Clone, Debug)]
pub struct CommitmentLedger {
    dir: PathBuf,
}

impl CommitmentLedger {
    /// Creates a fresh ledger. An existing state file, lock or temporary file
    /// makes initialization fail without changing anything.
    pub fn initialize(dir: &Path, policy: LedgerPolicy) -> Result<Self> {
        policy.validate()?;
        ensure_private_dir(dir)?;
        let _lock = acquire_lock(dir)?;
        if ledger_exists(dir) || dir.join(LEDGER_TEMP).exists() {
            let _ = release_lock(dir);
            return Err(Error::ArtifactAlreadyExists);
        }
        let state = LedgerState {
            version: COMMITMENT_LEDGER_VERSION,
            policy,
            seq: 0,
            head: Digest::ZERO,
            commitments: BTreeMap::new(),
            oracles: BTreeMap::new(),
            streams: BTreeMap::new(),
            requests: BTreeMap::new(),
            events: Vec::new(),
        };
        persist(dir, &state)?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    /// Opens an existing ledger and validates its state. A missing, damaged or
    /// rolled-back state file is rejected.
    pub fn open(dir: &Path) -> Result<Self> {
        ensure_private_dir(dir)?;
        if !ledger_exists(dir) {
            return Err(Error::ProverStateMissing);
        }
        let ledger = Self {
            dir: dir.to_path_buf(),
        };
        ledger.read_state()?;
        Ok(ledger)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Records the actual commitment once. Repeating the same value is
    /// idempotent; a different value for the same key is rejected.
    pub fn record_commitment(
        &self,
        kind: CommitmentKind,
        identity: Digest,
        commitment: Digest,
    ) -> Result<()> {
        let key = LedgerKey::new(kind, identity)?;
        if commitment == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        ensure_private_dir(&self.dir)?;
        let _lock = acquire_lock(&self.dir)?;
        if self.dir.join(LEDGER_TEMP).exists() {
            return abort(&self.dir, Error::StateLocked);
        }
        let mut state = match self.read_state() {
            Ok(state) => state,
            Err(error) => return abort(&self.dir, error),
        };
        match state.commitments.get_mut(&key) {
            Some(record) => {
                if record.commitment != commitment {
                    return abort(&self.dir, Error::StateRollbackSuspected);
                }
                return release_lock(&self.dir);
            }
            None => {
                state.commitments.insert(
                    key,
                    CommitmentRecord {
                        commitment,
                        logical_openings: 0,
                        table_reuse: 0,
                        attempts: BTreeMap::new(),
                    },
                );
            }
        }
        persist(&self.dir, &state)
    }

    /// Reserves one commitment-layer entry for one request or attempt before
    /// proof work. The reservation records the logical opening count, counts
    /// table reuse under the stored policy and returns a durable claim.
    #[must_use = "the reservation is already durable; dropping the claim does not release it"]
    pub fn reserve(
        &self,
        kind: CommitmentKind,
        identity: Digest,
        request: Digest,
        logical_openings: u64,
    ) -> Result<LedgerClaim> {
        let key = LedgerKey::new(kind, identity)?;
        if request == Digest::ZERO || logical_openings == 0 {
            return Err(Error::InvalidConfig);
        }
        ensure_private_dir(&self.dir)?;
        let _lock = acquire_lock(&self.dir)?;
        if self.dir.join(LEDGER_TEMP).exists() {
            return abort(&self.dir, Error::StateLocked);
        }
        let mut state = match self.read_state() {
            Ok(state) => state,
            Err(error) => return abort(&self.dir, error),
        };
        let table_reuse_policy = state.policy.table_reuse;
        let entry = state
            .commitments
            .entry(key)
            .or_insert_with(|| CommitmentRecord {
                // A caller that did not record a separate commitment binds
                // the root or identity itself. Recording a different value
                // later is rejected.
                commitment: identity,
                logical_openings: 0,
                table_reuse: 0,
                attempts: BTreeMap::new(),
            });
        if entry.attempts.contains_key(&request) {
            return abort(&self.dir, Error::LeafBudgetExhausted);
        }
        let reuse = match kind {
            CommitmentKind::Table => match table_reuse_policy {
                TableReusePolicy::Forbidden => {
                    if !entry.attempts.is_empty() {
                        return abort(&self.dir, Error::StateRollbackSuspected);
                    }
                    0
                }
                TableReusePolicy::Budget {
                    max_requests,
                    max_logical_openings,
                } => {
                    if entry.attempts.len() as u64 >= max_requests {
                        return abort(&self.dir, Error::LeafBudgetExhausted);
                    }
                    let charged = match entry.logical_openings.checked_add(logical_openings) {
                        Some(charged) => charged,
                        None => return abort(&self.dir, Error::ResourceLimit),
                    };
                    if charged > max_logical_openings {
                        return abort(&self.dir, Error::LeafBudgetExhausted);
                    }
                    entry.attempts.len() as u64
                }
            },
            CommitmentKind::Activation
            | CommitmentKind::Weight
            | CommitmentKind::Mask
            | CommitmentKind::Dot => {
                if !entry.attempts.is_empty() {
                    return abort(&self.dir, Error::StateRollbackSuspected);
                }
                0
            }
        };
        let charged = match entry.logical_openings.checked_add(logical_openings) {
            Some(charged) => charged,
            None => return abort(&self.dir, Error::ResourceLimit),
        };
        entry.logical_openings = charged;
        entry.table_reuse = reuse;
        entry.attempts.insert(
            request,
            AttemptRecord {
                state: AttemptState::Reserved,
                logical_openings,
            },
        );
        match persist(&self.dir, &state) {
            Ok(()) => Ok(LedgerClaim {
                dir: self.dir.clone(),
                key,
                request,
                logical_openings,
                table_reuse: reuse,
                attempt_state: AttemptState::Reserved,
            }),
            Err(error) => Err(error),
        }
    }

    /// Starts one opening-layer request and reserves its seed commitment before
    /// any proof work. The seed commitment is globally one-use; a second
    /// request with the same value is rejected even when its roots would
    /// differ.
    pub fn begin_request(
        &self,
        request: Digest,
        seed_commitment: Digest,
        plan_digest: Digest,
    ) -> Result<()> {
        if request == Digest::ZERO || seed_commitment == Digest::ZERO || plan_digest == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        self.update(|state| {
            if state.requests.contains_key(&request) {
                return Err(Error::LeafBudgetExhausted);
            }
            if state
                .requests
                .values()
                .any(|record| record.seed_commitment == seed_commitment)
            {
                return Err(Error::StateRollbackSuspected);
            }
            state.requests.insert(
                request,
                RequestRecord {
                    state: RequestState::Reserved,
                    seed_commitment,
                    plan_digest,
                    oracles: BTreeSet::new(),
                    events: Vec::new(),
                },
            );
            Ok(())
        })
    }

    /// Reserves one oracle material stream and its declared opening budget
    /// before proof work. The material stream is one-use per oracle; a table
    /// stream may serve the budgeted reuse of its own table oracle only.
    pub fn reserve_oracle(
        &self,
        request: Digest,
        oracle: OracleKey,
        stream: StreamKey,
        budget: OracleBudget,
    ) -> Result<()> {
        if oracle.identity == Digest::ZERO || stream.material == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        if !stream.matches(&oracle) {
            return Err(Error::InvalidConfig);
        }
        if budget.logical_draws == 0 && budget.allocated_rows == 0 && budget.committed_entries == 0
        {
            return Err(Error::InvalidConfig);
        }
        if budget.tail_rank != 0 && budget.logical_draws > budget.tail_rank {
            return Err(Error::LeafBudgetExhausted);
        }
        self.update(|state| {
            let request_record = state
                .requests
                .get(&request)
                .ok_or(Error::InvalidSessionState)?;
            if request_record.state != RequestState::Reserved {
                return Err(Error::InvalidSessionState);
            }
            let table_policy = state.policy.table_reuse;
            match state.oracles.get(&oracle) {
                None => {
                    if state.streams.contains_key(&stream) {
                        return Err(Error::StateRollbackSuspected);
                    }
                    state.streams.insert(
                        stream,
                        StreamRecord {
                            oracle,
                            first_request: request,
                        },
                    );
                    state.oracles.insert(
                        oracle,
                        OracleRecord {
                            commitment: None,
                            stream,
                            tail_rank: budget.tail_rank,
                            allocated_rows: budget.allocated_rows,
                            committed_entries: budget.committed_entries,
                            opened_positions: BTreeSet::new(),
                            logical_openings: 0,
                            response_entries: 0,
                            unique_positions: 0,
                            declared_logical: budget.logical_draws,
                            first_request: request,
                            requests: BTreeSet::from([request]),
                            per_stage: BTreeMap::new(),
                            state: EventState::Reserved,
                        },
                    );
                }
                Some(record) => {
                    if record.requests.contains(&request) {
                        return Err(Error::LeafBudgetExhausted);
                    }
                    if oracle.kind != OracleKind::Table {
                        return Err(Error::StateRollbackSuspected);
                    }
                    match table_policy {
                        TableReusePolicy::Forbidden => {
                            return Err(Error::StateRollbackSuspected);
                        }
                        TableReusePolicy::Budget {
                            max_requests,
                            max_logical_openings,
                        } => {
                            if record.requests.len() as u64 >= max_requests {
                                return Err(Error::LeafBudgetExhausted);
                            }
                            let declared = record
                                .declared_logical
                                .checked_add(budget.logical_draws)
                                .ok_or(Error::ResourceLimit)?;
                            if declared > max_logical_openings {
                                return Err(Error::LeafBudgetExhausted);
                            }
                        }
                    }
                    if record.stream != stream || record.tail_rank != budget.tail_rank {
                        return Err(Error::StateRollbackSuspected);
                    }
                    // The stream map already names this table oracle.
                    if state
                        .streams
                        .get(&stream)
                        .map(|entry| entry.oracle)
                        .ok_or(Error::StateRollbackSuspected)?
                        != oracle
                    {
                        return Err(Error::StateRollbackSuspected);
                    }
                    let record = state
                        .oracles
                        .get_mut(&oracle)
                        .ok_or(Error::StateRollbackSuspected)?;
                    record.requests.insert(request);
                    record.declared_logical = record
                        .declared_logical
                        .checked_add(budget.logical_draws)
                        .ok_or(Error::ResourceLimit)?;
                    record.allocated_rows = record
                        .allocated_rows
                        .checked_add(budget.allocated_rows)
                        .ok_or(Error::ResourceLimit)?;
                    record.committed_entries = record
                        .committed_entries
                        .checked_add(budget.committed_entries)
                        .ok_or(Error::ResourceLimit)?;
                }
            }
            state
                .requests
                .get_mut(&request)
                .ok_or(Error::InvalidSessionState)?
                .oracles
                .insert(oracle);
            Ok(())
        })
    }

    /// Binds the actual commitment root to a reserved oracle. The root may be
    /// bound once; a root already bound to another oracle or stream is
    /// rejected.
    pub fn bind_commitment(
        &self,
        request: Digest,
        oracle: OracleKey,
        commitment: Digest,
    ) -> Result<()> {
        if commitment == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        self.update(|state| {
            let request_record = state
                .requests
                .get(&request)
                .ok_or(Error::InvalidSessionState)?;
            if !request_record.oracles.contains(&oracle) {
                return Err(Error::InvalidSessionState);
            }
            let record = state
                .oracles
                .get(&oracle)
                .ok_or(Error::InvalidSessionState)?;
            match record.commitment {
                Some(existing) if existing == commitment => return Ok(()),
                Some(_) => return Err(Error::StatementMismatch),
                None => {}
            }
            for (other, entry) in &state.oracles {
                if *other != oracle && entry.commitment == Some(commitment) {
                    return Err(Error::StateRollbackSuspected);
                }
            }
            let record = state
                .oracles
                .get_mut(&oracle)
                .ok_or(Error::InvalidSessionState)?;
            record.commitment = Some(commitment);
            record.state = EventState::Bound;
            Ok(())
        })
    }

    /// Records one actual opening event. `positions` must be the distinct
    /// physical positions opened by the event; `counts` carries the logical
    /// draws and the proof response entries separately. The event is rejected
    /// when the union of distinct positions on this tree would exceed the code
    /// tail rank or when the cumulative logical draws would exceed the
    /// declared budget; the request is then closed as failed and the material
    /// stays spent.
    pub fn record_openings(
        &self,
        request: Digest,
        oracle: OracleKey,
        stage: Stage,
        source: OpeningSource,
        counts: OpeningCounts,
        positions: &[u64],
    ) -> Result<TailState> {
        if counts.unique_positions != positions.len() as u64 {
            return Err(Error::InvalidConfig);
        }
        OpeningCounts::new(
            counts.logical_draws,
            counts.response_entries,
            counts.unique_positions,
        )?;
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != positions.len() {
            return Err(Error::InvalidConfig);
        }
        ensure_private_dir(&self.dir)?;
        let _lock = acquire_lock(&self.dir)?;
        if self.dir.join(LEDGER_TEMP).exists() {
            return abort(&self.dir, Error::StateLocked);
        }
        let mut state = match read_ledger_state(&self.dir) {
            Ok(state) => state,
            Err(error) => return abort(&self.dir, error),
        };
        let request_record = match state.requests.get(&request) {
            Some(record) => record,
            None => return abort(&self.dir, Error::InvalidSessionState),
        };
        if request_record.state != RequestState::Reserved {
            return abort(&self.dir, Error::InvalidSessionState);
        }
        let record = match state.oracles.get(&oracle) {
            Some(record) => record,
            None => return abort(&self.dir, Error::InvalidSessionState),
        };
        if !record.requests.contains(&request) {
            return abort(&self.dir, Error::InvalidSessionState);
        }
        let prior_union = record.opened_positions.len() as u64;
        let charged_logical = match record.logical_openings.checked_add(counts.logical_draws) {
            Some(charged) => charged,
            None => return abort(&self.dir, Error::ResourceLimit),
        };
        let mut union = record.opened_positions.clone();
        for position in &sorted {
            union.insert(*position);
        }
        let new_union = union.len() as u64;
        if (record.tail_rank != 0 && new_union > record.tail_rank)
            || charged_logical > record.declared_logical
        {
            // Fail closed: persist the aborted reservation so a caller cannot
            // catch the error and continue with different positions.
            let request_record = state
                .requests
                .get_mut(&request)
                .ok_or(Error::InvalidSessionState)?;
            request_record.state = RequestState::Failed;
            let record = state
                .oracles
                .get_mut(&oracle)
                .ok_or(Error::InvalidSessionState)?;
            record.state = EventState::Aborted;
            if let Err(error) = persist(&self.dir, &state) {
                return Err(error);
            }
            return Err(Error::LeafBudgetExhausted);
        }
        let remaining = if record.tail_rank == 0 {
            0
        } else {
            record
                .tail_rank
                .checked_sub(new_union)
                .ok_or(Error::StateRollbackSuspected)?
        };
        let attempt = state
            .requests
            .get(&request)
            .ok_or(Error::InvalidSessionState)?
            .events
            .len() as u64
            + 1;
        let prev_event = state.head;
        let seq = state.seq.checked_add(1).ok_or(Error::ResourceLimit)?;
        let mut event = LedgerEvent {
            schema: LEDGER_EVENT_SCHEMA_VERSION,
            seq,
            prev_event,
            event_id: Digest::ZERO,
            request,
            attempt,
            oracle,
            stream: record.stream,
            stage,
            source,
            counts,
            tail: TailState {
                rank: record.tail_rank,
                prior_union,
                new_union,
                remaining,
            },
            positions: sorted,
            state: EventState::Opened,
        };
        event.event_id = event.compute_id();
        let event_id = event.event_id;
        let tail = event.tail;
        let oracle_record = state
            .oracles
            .get_mut(&oracle)
            .ok_or(Error::InvalidSessionState)?;
        oracle_record.opened_positions = union;
        oracle_record.logical_openings = oracle_record
            .logical_openings
            .checked_add(counts.logical_draws)
            .ok_or(Error::ResourceLimit)?;
        oracle_record.response_entries = oracle_record
            .response_entries
            .checked_add(counts.response_entries)
            .ok_or(Error::ResourceLimit)?;
        oracle_record.unique_positions = new_union;
        oracle_record.state = EventState::Opened;
        oracle_record
            .per_stage
            .entry(stage)
            .or_insert_with(OpeningCounts::default)
            .add_into(counts)?;
        state
            .requests
            .get_mut(&request)
            .ok_or(Error::InvalidSessionState)?
            .events
            .push(seq);
        state.seq = seq;
        state.head = event_id;
        state.events.push(event);
        persist(&self.dir, &state)?;
        Ok(tail)
    }

    /// Advances the request lifecycle. Only `Reserved -> Proved ->
    /// Verified -> Succeeded` is legal; `Failed` is allowed from any
    /// non-terminal state and is terminal. Success before verification is
    /// rejected.
    pub fn finish_request(&self, request: Digest, outcome: RequestOutcome) -> Result<()> {
        let next = match outcome {
            RequestOutcome::Proved => RequestState::Proved,
            RequestOutcome::Verified => RequestState::Verified,
            RequestOutcome::Succeeded => RequestState::Succeeded,
            RequestOutcome::Failed => RequestState::Failed,
        };
        self.update(|state| {
            let record = state
                .requests
                .get_mut(&request)
                .ok_or(Error::InvalidSessionState)?;
            let legal = match next {
                RequestState::Proved => record.state == RequestState::Reserved,
                RequestState::Verified => record.state == RequestState::Proved,
                RequestState::Succeeded => record.state == RequestState::Verified,
                RequestState::Failed => matches!(
                    record.state,
                    RequestState::Reserved | RequestState::Proved | RequestState::Verified
                ),
                RequestState::Reserved => false,
            };
            if !legal {
                return Err(Error::InvalidSessionState);
            }
            record.state = next;
            Ok(())
        })
    }

    /// Records that the prover returned. The request stays unverified.
    pub fn mark_proved(&self, request: Digest) -> Result<()> {
        self.finish_request(request, RequestOutcome::Proved)
    }

    /// Records that the complete verification accepted.
    pub fn mark_verified(&self, request: Digest) -> Result<()> {
        self.finish_request(request, RequestOutcome::Verified)
    }

    /// Closes a verified request. It is rejected before verification.
    pub fn succeed_request(&self, request: Digest) -> Result<()> {
        self.finish_request(request, RequestOutcome::Succeeded)
    }

    /// Closes a request as failed. The material stays spent forever.
    pub fn fail_request(&self, request: Digest) -> Result<()> {
        self.finish_request(request, RequestOutcome::Failed)
    }

    /// Reads and validates the authoritative state. A concurrent writer's
    /// temporary file is ignored; the state file is replaced atomically.
    pub fn read_state(&self) -> Result<LedgerState> {
        let state: LedgerState = decode_bytes(&read_bytes(&self.dir.join(LEDGER_FILE))?)?;
        state.validate()?;
        Ok(state)
    }

    /// Reads one commitment-layer entry from a validated snapshot.
    pub fn record_for(
        &self,
        kind: CommitmentKind,
        identity: Digest,
    ) -> Result<Option<CommitmentRecord>> {
        let key = LedgerKey::new(kind, identity)?;
        Ok(self.read_state()?.record(key).cloned())
    }

    /// Reads one oracle record from a validated snapshot.
    pub fn oracle_record(&self, oracle: OracleKey) -> Result<Option<OracleRecord>> {
        Ok(self.read_state()?.oracles.get(&oracle).cloned())
    }

    /// Reads one request record from a validated snapshot.
    pub fn request_record(&self, request: Digest) -> Result<Option<RequestRecord>> {
        Ok(self.read_state()?.requests.get(&request).cloned())
    }

    /// The append-only opening-event log.
    pub fn events(&self) -> Result<Vec<LedgerEvent>> {
        Ok(self.read_state()?.events)
    }

    fn update<T>(&self, apply: impl FnOnce(&mut LedgerState) -> Result<T>) -> Result<T> {
        ensure_private_dir(&self.dir)?;
        let _lock = acquire_lock(&self.dir)?;
        if self.dir.join(LEDGER_TEMP).exists() {
            return abort(&self.dir, Error::StateLocked);
        }
        let mut state = match read_ledger_state(&self.dir) {
            Ok(state) => state,
            Err(error) => return abort(&self.dir, error),
        };
        let outcome = match apply(&mut state) {
            Ok(outcome) => outcome,
            Err(error) => return abort(&self.dir, error),
        };
        persist(&self.dir, &state)?;
        Ok(outcome)
    }
}

fn ledger_exists(dir: &Path) -> bool {
    dir.join(LEDGER_FILE).exists()
}

fn acquire_lock(dir: &Path) -> Result<File> {
    let path = dir.join(LEDGER_LOCK);
    for _ in 0..LOCK_ATTEMPTS {
        match private_options().open(&path) {
            Ok(file) => {
                // The lock must be durable before the state write begins.
                file.sync_all().map_err(|_| Error::Io)?;
                sync_dir(dir)?;
                return Ok(file);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(LOCK_RETRY);
            }
            Err(_) => return Err(Error::Io),
        }
    }
    Err(Error::StateLocked)
}

fn release_lock(dir: &Path) -> Result<()> {
    std::fs::remove_file(dir.join(LEDGER_LOCK)).map_err(|_| Error::Io)?;
    sync_dir(dir)
}

/// Releases the lock after a rejection that did not write any new state. If
/// the release itself fails, the stale lock keeps every later reservation
/// closed.
fn abort<T>(dir: &Path, error: Error) -> Result<T> {
    let _ = release_lock(dir);
    Err(error)
}

/// Writes and publishes the state. A failure before the rename leaves the old
/// state and releases the lock; a failure after the rename leaves the lock in
/// place, so a retry fails closed.
fn persist(dir: &Path, state: &LedgerState) -> Result<()> {
    state.validate()?;
    let temp = dir.join(LEDGER_TEMP);
    let path = dir.join(LEDGER_FILE);
    let bytes = encode_bytes(state)?;
    if write_bytes(&temp, &bytes, true).is_err() {
        let _ = std::fs::remove_file(&temp);
        let _ = release_lock(dir);
        return Err(Error::Io);
    }
    if std::fs::rename(&temp, &path).is_err() {
        let _ = std::fs::remove_file(&temp);
        let _ = release_lock(dir);
        return Err(Error::Io);
    }
    // The new state is visible. Every later failure keeps the lock in place.
    sync_dir(dir)?;
    release_lock(dir)
}

fn finish_attempt(dir: &Path, key: LedgerKey, request: Digest, state: AttemptState) -> Result<()> {
    if state == AttemptState::Reserved {
        return Err(Error::InvalidConfig);
    }
    ensure_private_dir(dir)?;
    let _lock = acquire_lock(dir)?;
    if dir.join(LEDGER_TEMP).exists() {
        return abort(dir, Error::StateLocked);
    }
    let mut ledger = match read_ledger_state(dir) {
        Ok(state) => state,
        Err(error) => return abort(dir, error),
    };
    let record = match ledger.commitments.get_mut(&key) {
        Some(record) => record,
        None => return abort(dir, Error::InvalidSessionState),
    };
    let attempt = match record.attempts.get_mut(&request) {
        Some(attempt) => attempt,
        None => return abort(dir, Error::InvalidSessionState),
    };
    if attempt.state != AttemptState::Reserved {
        return abort(dir, Error::InvalidSessionState);
    }
    attempt.state = state;
    persist(dir, &ledger)
}

fn read_ledger_state(dir: &Path) -> Result<LedgerState> {
    let state: LedgerState = decode_bytes(&read_bytes(&dir.join(LEDGER_FILE))?)?;
    state.validate()?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    use super::*;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zkie-commitment-ledger-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn root(seed: u8) -> Digest {
        Digest([seed; 32])
    }

    fn request(seed: u8) -> Digest {
        Digest([seed; 32])
    }

    fn key(kind: CommitmentKind, identity: Digest) -> LedgerKey {
        LedgerKey::new(kind, identity).expect("key")
    }

    fn digest_hex(value: Digest) -> String {
        value.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn digest_from_hex(text: &str) -> Digest {
        assert_eq!(text.len(), 64);
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex");
        }
        Digest(bytes)
    }

    #[test]
    fn initialization_is_fresh_and_rejects_a_second_state() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        assert!(CommitmentLedger::initialize(&dir, LedgerPolicy::default()).is_err());
        assert_eq!(ledger.read_state().expect("state").commitments.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_actual_commitment_is_recorded_once_and_conflicts_are_rejected() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(1);
        ledger
            .record_commitment(CommitmentKind::Table, identity, root(2))
            .expect("record");
        assert!(ledger
            .record_commitment(CommitmentKind::Table, identity, root(3))
            .is_err());
        let record = ledger
            .record_for(CommitmentKind::Table, identity)
            .expect("read")
            .expect("record");
        assert_eq!(record.commitment, root(2));
        assert_eq!(record.attempts.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reservation_records_the_attempt_and_the_logical_opening_count() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(4);
        let attempt = request(5);
        let claim = ledger
            .reserve(CommitmentKind::Activation, identity, attempt, 7)
            .expect("reserve");
        assert_eq!(claim.logical_openings(), 7);
        assert_eq!(claim.table_reuse(), 0);
        assert_eq!(claim.attempt_state(), AttemptState::Reserved);
        let state = ledger.read_state().expect("state");
        let record = state
            .record(key(CommitmentKind::Activation, identity))
            .expect("record");
        assert_eq!(record.logical_openings, 7);
        assert_eq!(
            record.attempt(attempt).expect("attempt").state,
            AttemptState::Reserved
        );
        claim.succeed().expect("succeed");
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state
                .record(key(CommitmentKind::Activation, identity))
                .expect("record")
                .attempt(attempt)
                .expect("attempt")
                .state,
            AttemptState::Succeeded
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_activation_root_cannot_serve_a_second_request() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(6);
        let first = ledger
            .reserve(CommitmentKind::Activation, identity, request(7), 3)
            .expect("first");
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, identity, request(8), 3)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        first.succeed().expect("succeed");
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, identity, request(9), 3)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        let state = ledger.read_state().expect("state");
        let record = state
            .record(key(CommitmentKind::Activation, identity))
            .expect("record");
        assert_eq!(record.logical_openings, 3);
        assert_eq!(record.attempts.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_attempt_cannot_be_retried() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(10);
        let attempt = request(11);
        let claim = ledger
            .reserve(CommitmentKind::Weight, identity, attempt, 2)
            .expect("reserve");
        claim.fail().expect("fail");
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Weight, identity, attempt, 2)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Weight, identity, request(12), 2)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state
                .record(key(CommitmentKind::Weight, identity))
                .expect("record")
                .attempt(attempt)
                .expect("attempt")
                .state,
            AttemptState::Failed
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn table_reuse_is_forbidden_or_counted_by_policy() {
        let forbidden_dir = temp_dir();
        let forbidden = CommitmentLedger::initialize(&forbidden_dir, LedgerPolicy::default())
            .expect("init forbidden");
        let table = root(13);
        forbidden
            .record_commitment(CommitmentKind::Table, table, root(14))
            .expect("record");
        let first_forbidden = forbidden
            .reserve(CommitmentKind::Table, table, request(15), 603)
            .expect("first");
        assert_eq!(first_forbidden.table_reuse(), 0);
        first_forbidden.succeed().expect("succeed");
        assert_eq!(
            forbidden
                .reserve(CommitmentKind::Table, table, request(16), 603)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );

        let budget_dir = temp_dir();
        let policy = LedgerPolicy {
            table_reuse: TableReusePolicy::Budget {
                max_requests: 2,
                max_logical_openings: 1_206,
            },
        };
        let budget = CommitmentLedger::initialize(&budget_dir, policy).expect("init budget");
        budget
            .record_commitment(CommitmentKind::Table, table, root(14))
            .expect("record");
        let first = budget
            .reserve(CommitmentKind::Table, table, request(17), 603)
            .expect("first");
        assert_eq!(first.table_reuse(), 0);
        first.succeed().expect("succeed");
        let second = budget
            .reserve(CommitmentKind::Table, table, request(18), 603)
            .expect("reuse");
        assert_eq!(second.table_reuse(), 1);
        second.succeed().expect("succeed");
        let state = budget.read_state().expect("state");
        let record = state
            .record(key(CommitmentKind::Table, table))
            .expect("record");
        assert_eq!(record.logical_openings, 1_206);
        assert_eq!(record.table_reuse_count(), 1);
        assert_eq!(record.attempts.len(), 2);
        assert_eq!(state.total_logical_openings().expect("total"), 1_206);
        assert_eq!(state.total_table_reuse().expect("reuse"), 1);
        assert_eq!(
            budget
                .reserve(CommitmentKind::Table, table, request(19), 1)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let after = budget.read_state().expect("state");
        assert_eq!(
            after
                .record(key(CommitmentKind::Table, table))
                .expect("record")
                .logical_openings,
            1_206
        );
        let _ = std::fs::remove_dir_all(&forbidden_dir);
        let _ = std::fs::remove_dir_all(&budget_dir);
    }

    #[test]
    fn a_stale_lock_fails_closed_without_consuming_the_ledger() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        std::fs::write(dir.join(LEDGER_LOCK), b"stale").expect("lock");
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, root(20), request(21), 1)
                .unwrap_err(),
            Error::StateLocked
        );
        assert_eq!(ledger.read_state().expect("state").commitments.len(), 0);
        std::fs::remove_file(dir.join(LEDGER_LOCK)).expect("remove lock");
        let recovered = ledger
            .reserve(CommitmentKind::Activation, root(20), request(21), 1)
            .expect("reserve after recovery");
        recovered.succeed().expect("succeed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_temporary_fails_closed_without_consuming_the_ledger() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        std::fs::write(dir.join(LEDGER_TEMP), b"stale").expect("temp");
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, root(22), request(23), 1)
                .unwrap_err(),
            Error::StateLocked
        );
        assert_eq!(ledger.read_state().expect("state").commitments.len(), 0);
        std::fs::remove_file(dir.join(LEDGER_TEMP)).expect("remove temp");
        let recovered = ledger
            .reserve(CommitmentKind::Activation, root(22), request(23), 1)
            .expect("reserve after recovery");
        recovered.succeed().expect("succeed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_threads_have_one_winner() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(24);
        let attempt = request(25);
        let first = ledger.clone();
        let second = ledger.clone();
        let a = thread::spawn(move || {
            first
                .reserve(CommitmentKind::Activation, identity, attempt, 9)
                .is_ok()
        });
        let b = thread::spawn(move || {
            second
                .reserve(CommitmentKind::Activation, identity, attempt, 9)
                .is_ok()
        });
        let wins = [a.join().expect("join"), b.join().expect("join")]
            .into_iter()
            .filter(|won| *won)
            .count();
        assert_eq!(wins, 1);
        let state = ledger.read_state().expect("state");
        let record = state
            .record(key(CommitmentKind::Activation, identity))
            .expect("record");
        assert_eq!(record.attempts.len(), 1);
        assert_eq!(record.logical_openings, 9);
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, identity, attempt, 9)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn spawn_worker(
        dir: &Path,
        kind: CommitmentKind,
        identity: Digest,
        attempt: Digest,
        logical_openings: u64,
        finish: &str,
    ) -> Child {
        let executable = std::env::current_exe().expect("test executable");
        let kind = match kind {
            CommitmentKind::Activation => "activation",
            CommitmentKind::Weight => "weight",
            CommitmentKind::Table => "table",
            CommitmentKind::Mask => "mask",
            CommitmentKind::Dot => "dot",
        };
        Command::new(executable)
            .args([
                "--exact",
                "commitment_ledger::tests::commitment_ledger_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("ZKIE_LEDGER_WORKER_DIR", dir)
            .env("ZKIE_LEDGER_WORKER_KIND", kind)
            .env("ZKIE_LEDGER_WORKER_IDENTITY", digest_hex(identity))
            .env("ZKIE_LEDGER_WORKER_REQUEST", digest_hex(attempt))
            .env("ZKIE_LEDGER_WORKER_OPENINGS", logical_openings.to_string())
            .env("ZKIE_LEDGER_WORKER_FINISH", finish)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn ledger worker")
    }

    #[test]
    #[ignore]
    fn commitment_ledger_worker() {
        let Ok(dir) = std::env::var("ZKIE_LEDGER_WORKER_DIR") else {
            return;
        };
        let kind = match std::env::var("ZKIE_LEDGER_WORKER_KIND").as_deref() {
            Ok("weight") => CommitmentKind::Weight,
            Ok("table") => CommitmentKind::Table,
            Ok("mask") => CommitmentKind::Mask,
            _ => CommitmentKind::Activation,
        };
        let identity =
            digest_from_hex(&std::env::var("ZKIE_LEDGER_WORKER_IDENTITY").expect("identity"));
        let attempt =
            digest_from_hex(&std::env::var("ZKIE_LEDGER_WORKER_REQUEST").expect("request"));
        let logical_openings: u64 = std::env::var("ZKIE_LEDGER_WORKER_OPENINGS")
            .expect("openings")
            .parse()
            .expect("openings number");
        let finish =
            std::env::var("ZKIE_LEDGER_WORKER_FINISH").unwrap_or_else(|_| "none".to_string());
        let ledger = CommitmentLedger::open(Path::new(&dir)).expect("open ledger");
        match ledger.reserve(kind, identity, attempt, logical_openings) {
            Ok(claim) => match finish.as_str() {
                "succeed" => claim.succeed().expect("succeed"),
                "fail" => claim.fail().expect("fail"),
                // A process that exits here models a crash after the durable
                // reservation and before the proof outcome is recorded.
                _ => std::process::exit(0),
            },
            Err(error) => {
                eprintln!("reservation rejected: {error:?}");
                std::process::exit(3);
            }
        }
    }

    #[test]
    fn concurrent_processes_have_one_winner() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(26);
        let attempt = request(27);
        let mut first = spawn_worker(
            &dir,
            CommitmentKind::Activation,
            identity,
            attempt,
            5,
            "none",
        );
        let mut second = spawn_worker(
            &dir,
            CommitmentKind::Activation,
            identity,
            attempt,
            5,
            "none",
        );
        let first_status = first.wait().expect("wait first");
        let second_status = second.wait().expect("wait second");
        let wins = usize::from(first_status.success()) + usize::from(second_status.success());
        assert_eq!(wins, 1);
        let state = ledger.read_state().expect("state");
        let record = state
            .record(key(CommitmentKind::Activation, identity))
            .expect("record");
        assert_eq!(record.attempts.len(), 1);
        assert_eq!(record.logical_openings, 5);
        assert_eq!(
            ledger
                .reserve(CommitmentKind::Activation, identity, attempt, 5)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_restart_sees_the_spent_state_and_cannot_reuse_it() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(28);
        let attempt = request(29);
        let status = spawn_worker(&dir, CommitmentKind::Weight, identity, attempt, 4, "none")
            .wait()
            .expect("wait");
        assert!(status.success());
        let reopened = CommitmentLedger::open(&dir).expect("reopen");
        let state = reopened.read_state().expect("state");
        assert_eq!(
            state
                .record(key(CommitmentKind::Weight, identity))
                .expect("record")
                .attempt(attempt)
                .expect("attempt")
                .state,
            AttemptState::Reserved
        );
        assert_eq!(
            reopened
                .reserve(CommitmentKind::Weight, identity, attempt, 4)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        assert_eq!(
            reopened
                .reserve(CommitmentKind::Weight, identity, request(30), 4)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        drop(ledger);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_attempt_in_another_process_cannot_reuse_its_reservation() {
        let dir = temp_dir();
        let _ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let identity = root(31);
        let attempt = request(32);
        let status = spawn_worker(
            &dir,
            CommitmentKind::Activation,
            identity,
            attempt,
            2,
            "fail",
        )
        .wait()
        .expect("wait");
        assert!(status.success());
        let reopened = CommitmentLedger::open(&dir).expect("reopen");
        assert_eq!(
            reopened
                .reserve(CommitmentKind::Activation, identity, attempt, 2)
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_state_is_rejected() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        std::fs::write(dir.join(LEDGER_FILE), b"damaged").expect("damage");
        assert!(ledger.read_state().is_err());
        assert!(ledger
            .reserve(CommitmentKind::Activation, root(33), request(34), 1)
            .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // Opening-event layer
    // -----------------------------------------------------------------------

    fn digest_u64(value: u64) -> Digest {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&value.to_le_bytes());
        Digest(bytes)
    }

    fn oracle(
        kind: OracleKind,
        identity_seed: u64,
        oracle_id: u64,
        tree_number: u64,
        run_seed: u64,
        configuration_seed: u64,
    ) -> OracleKey {
        OracleKey::new(
            kind,
            digest_u64(identity_seed),
            oracle_id,
            tree_number,
            digest_u64(run_seed),
            digest_u64(configuration_seed),
        )
        .expect("oracle key")
    }

    fn stream_for(oracle: &OracleKey, material_seed: u64) -> StreamKey {
        StreamKey::new(
            oracle.kind,
            oracle.oracle_id,
            oracle.tree_number,
            oracle.run,
            oracle.configuration,
            digest_u64(material_seed),
        )
        .expect("stream key")
    }

    fn budget(logical_draws: u64, tail_rank: u64) -> OracleBudget {
        OracleBudget::new(logical_draws, tail_rank).expect("budget")
    }

    fn counts(logical: u64, response: u64, unique: u64) -> OpeningCounts {
        OpeningCounts::new(logical, response, unique).expect("counts")
    }

    fn range(start: u64, count: u64) -> Vec<u64> {
        (start..start + count).collect()
    }

    #[test]
    fn the_same_seed_commitment_across_requests_is_rejected() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let seed = root(40);
        ledger
            .begin_request(request(41), seed, root(42))
            .expect("first request");
        assert_eq!(
            ledger
                .begin_request(request(43), seed, root(44))
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        let state = ledger.read_state().expect("state");
        assert_eq!(state.requests.len(), 1);
        assert!(state.request(request(41)).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_streams_with_equal_positions_or_values_are_accepted() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let first_request = request(45);
        let second_request = request(46);
        ledger
            .begin_request(first_request, root(47), root(48))
            .expect("first request");
        ledger
            .begin_request(second_request, root(49), root(50))
            .expect("second request");
        let first_oracle = oracle(OracleKind::Weight, 51, 20, 100, 52, 53);
        let second_oracle = oracle(OracleKind::Weight, 54, 21, 101, 52, 53);
        let first_stream = stream_for(&first_oracle, 55);
        let second_stream = stream_for(&second_oracle, 56);
        // The two streams are independent even though they draw the same
        // positions and the same field values. Position or value equality is
        // not a rejection reason.
        for (request, oracle, stream) in [
            (first_request, first_oracle, first_stream),
            (second_request, second_oracle, second_stream),
        ] {
            ledger
                .reserve_oracle(request, oracle, stream, budget(1, 8))
                .expect("reserve");
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Send,
                    OpeningSource::SendSample,
                    counts(1, 1, 1),
                    &[7],
                )
                .expect("opening");
        }
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state.oracle(first_oracle).expect("first").opened_positions,
            BTreeSet::from([7])
        );
        assert_eq!(
            state
                .oracle(second_oracle)
                .expect("second")
                .opened_positions,
            BTreeSet::from([7])
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_material_stream_across_requests_is_rejected_even_with_another_root() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let first_request = request(60);
        let second_request = request(61);
        ledger
            .begin_request(first_request, root(62), root(63))
            .expect("first request");
        ledger
            .begin_request(second_request, root(64), root(65))
            .expect("second request");
        let oracle = oracle(OracleKind::Weight, 66, 20, 100, 67, 68);
        let stream = stream_for(&oracle, 69);
        ledger
            .reserve_oracle(first_request, oracle, stream, budget(1, 8))
            .expect("first material");
        ledger
            .bind_commitment(first_request, oracle, root(70))
            .expect("first root");
        // A second request that tries to regenerate the same material with a
        // fresh salt and therefore a different root is rejected. A different
        // root does not imply different material.
        assert_eq!(
            ledger
                .reserve_oracle(second_request, oracle, stream, budget(1, 8))
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        assert!(ledger
            .bind_commitment(second_request, oracle, root(71))
            .is_err());
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state.oracle(oracle).expect("oracle").commitment,
            Some(root(70))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_commitment_root_under_another_stream_is_rejected() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let first_request = request(72);
        let second_request = request(73);
        ledger
            .begin_request(first_request, root(74), root(75))
            .expect("first request");
        ledger
            .begin_request(second_request, root(76), root(77))
            .expect("second request");
        let first_oracle = oracle(OracleKind::Weight, 78, 20, 100, 79, 80);
        let second_oracle = oracle(OracleKind::Weight, 81, 21, 101, 79, 80);
        ledger
            .reserve_oracle(
                first_request,
                first_oracle,
                stream_for(&first_oracle, 82),
                budget(1, 8),
            )
            .expect("first reserve");
        ledger
            .reserve_oracle(
                second_request,
                second_oracle,
                stream_for(&second_oracle, 83),
                budget(1, 8),
            )
            .expect("second reserve");
        ledger
            .bind_commitment(first_request, first_oracle, root(84))
            .expect("first root");
        assert_eq!(
            ledger
                .bind_commitment(second_request, second_oracle, root(84))
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .oracle_record(second_oracle)
                .expect("record")
                .expect("oracle")
                .commitment,
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tail_union_accumulates_across_stages_and_an_overrun_fails_closed() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(90);
        ledger
            .begin_request(request, root(91), root(92))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 93, 20, 100, 94, 95);
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 96), budget(2, 2))
            .expect("reserve");
        let send = ledger
            .record_openings(
                request,
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(1, 1, 1),
                &[1],
            )
            .expect("send union");
        assert_eq!(send.prior_union, 0);
        assert_eq!(send.new_union, 1);
        assert_eq!(send.remaining, 1);
        let base = ledger
            .record_openings(
                request,
                oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                counts(1, 1, 1),
                &[2],
            )
            .expect("base union");
        assert_eq!(base.prior_union, 1);
        assert_eq!(base.new_union, 2);
        assert_eq!(base.remaining, 0);
        assert_eq!(
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Second,
                    OpeningSource::BaseMember,
                    counts(1, 1, 1),
                    &[3],
                )
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        // The overrun is persisted: the request is failed, the oracle is
        // aborted, the material stays spent, and the prior union is intact.
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state.request(request).expect("request").state,
            RequestState::Failed
        );
        let record = state.oracle(oracle).expect("oracle");
        assert_eq!(record.state, EventState::Aborted);
        assert_eq!(record.opened_positions, BTreeSet::from([1, 2]));
        assert_eq!(record.logical_openings, 2);
        assert_eq!(
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Second,
                    OpeningSource::BaseMember,
                    counts(1, 1, 1),
                    &[4],
                )
                .unwrap_err(),
            Error::InvalidSessionState
        );
        assert_eq!(
            ledger.mark_proved(request).unwrap_err(),
            Error::InvalidSessionState
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn logical_draws_count_repeats_while_the_tail_union_deduplicates() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(97);
        ledger
            .begin_request(request, root(98), root(99))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 100, 20, 100, 101, 102);
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 103), budget(3, 4))
            .expect("reserve");
        // Two logical draws of the same column produce one response entry and
        // one distinct position, but they are charged twice.
        let first = ledger
            .record_openings(
                request,
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(2, 1, 1),
                &[7],
            )
            .expect("repeat draw");
        assert_eq!(first.new_union, 1);
        let second = ledger
            .record_openings(
                request,
                oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                counts(1, 1, 1),
                &[8],
            )
            .expect("next draw");
        assert_eq!(second.new_union, 2);
        let record = ledger.oracle_record(oracle).expect("read").expect("oracle");
        assert_eq!(record.logical_openings, 3);
        assert_eq!(record.response_entries, 2);
        assert_eq!(record.unique_positions, 2);
        assert_eq!(record.opened_positions, BTreeSet::from([7, 8]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_declared_budget_above_the_tail_rank_is_rejected_before_any_event() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(104);
        ledger
            .begin_request(request, root(105), root(106))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 107, 20, 100, 108, 109);
        assert_eq!(
            ledger
                .reserve_oracle(
                    request,
                    oracle,
                    stream_for(&oracle, 110),
                    OracleBudget {
                        logical_draws: 5,
                        tail_rank: 4,
                        allocated_rows: 0,
                        committed_entries: 0,
                    },
                )
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let state = ledger.read_state().expect("state");
        assert!(state.oracle(oracle).is_none());
        assert_eq!(state.streams.len(), 0);
        assert_eq!(state.events.len(), 0);
        assert_eq!(
            state.request(request).expect("request").state,
            RequestState::Reserved
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn spawn_opening_worker(
        dir: &Path,
        request: Digest,
        material: Digest,
        position: u64,
        finish: &str,
    ) -> Child {
        let executable = std::env::current_exe().expect("test executable");
        Command::new(executable)
            .args([
                "--exact",
                "commitment_ledger::tests::opening_ledger_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("ZKIE_OPENING_WORKER_DIR", dir)
            .env("ZKIE_OPENING_WORKER_REQUEST", digest_hex(request))
            .env("ZKIE_OPENING_WORKER_MATERIAL", digest_hex(material))
            .env("ZKIE_OPENING_WORKER_POSITION", position.to_string())
            .env("ZKIE_OPENING_WORKER_FINISH", finish)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn opening worker")
    }

    fn worker_oracle() -> OracleKey {
        oracle(OracleKind::Weight, 0x90, 20, 100, 0x91, 0x92)
    }

    fn worker_stream() -> StreamKey {
        stream_for(&worker_oracle(), 0x93)
    }

    #[test]
    #[ignore]
    fn opening_ledger_worker() {
        let Ok(dir) = std::env::var("ZKIE_OPENING_WORKER_DIR") else {
            return;
        };
        let request =
            digest_from_hex(&std::env::var("ZKIE_OPENING_WORKER_REQUEST").expect("request"));
        let _material =
            digest_from_hex(&std::env::var("ZKIE_OPENING_WORKER_MATERIAL").expect("material"));
        let position: u64 = std::env::var("ZKIE_OPENING_WORKER_POSITION")
            .expect("position")
            .parse()
            .expect("position number");
        let finish =
            std::env::var("ZKIE_OPENING_WORKER_FINISH").unwrap_or_else(|_| "none".to_string());
        let ledger = CommitmentLedger::open(Path::new(&dir)).expect("open ledger");
        let oracle = worker_oracle();
        ledger
            .reserve_oracle(request, oracle, worker_stream(), budget(1, 8))
            .expect("reserve");
        ledger
            .record_openings(
                request,
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(1, 1, 1),
                &[position],
            )
            .expect("record");
        match finish.as_str() {
            "proved" => ledger.mark_proved(request).expect("proved"),
            "verified" => {
                ledger.mark_proved(request).expect("proved");
                ledger.mark_verified(request).expect("verified");
            }
            "succeeded" => {
                ledger.mark_proved(request).expect("proved");
                ledger.mark_verified(request).expect("verified");
                ledger.succeed_request(request).expect("succeeded");
            }
            "fail" => ledger.fail_request(request).expect("fail"),
            // A process that exits here models a crash after the durable
            // material reservation and opening and before any outcome.
            _ => std::process::exit(0),
        }
    }

    #[test]
    fn a_crash_after_the_reservation_blocks_retry_and_material_reuse() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request_id = request(120);
        ledger
            .begin_request(request_id, root(121), root(122))
            .expect("begin");
        let status = spawn_opening_worker(&dir, request_id, root(123), 11, "none")
            .wait()
            .expect("wait");
        assert!(status.success());
        let reopened = CommitmentLedger::open(&dir).expect("reopen");
        let state = reopened.read_state().expect("state");
        assert_eq!(
            state.request(request_id).expect("request").state,
            RequestState::Reserved
        );
        assert_eq!(
            state
                .oracle(worker_oracle())
                .expect("oracle")
                .opened_positions,
            BTreeSet::from([11])
        );
        assert_eq!(
            reopened
                .reserve_oracle(request_id, worker_oracle(), worker_stream(), budget(1, 8))
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let second = request(124);
        reopened
            .begin_request(second, root(125), root(126))
            .expect("second request");
        assert_eq!(
            reopened
                .reserve_oracle(second, worker_oracle(), worker_stream(), budget(1, 8))
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn success_requires_the_complete_verification() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(130);
        ledger
            .begin_request(request, root(131), root(132))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 133, 20, 100, 134, 135);
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 136), budget(1, 8))
            .expect("reserve");
        ledger
            .record_openings(
                request,
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(1, 1, 1),
                &[5],
            )
            .expect("opening");
        assert_eq!(
            ledger.mark_verified(request).unwrap_err(),
            Error::InvalidSessionState
        );
        assert_eq!(
            ledger.succeed_request(request).unwrap_err(),
            Error::InvalidSessionState
        );
        ledger.mark_proved(request).expect("proved");
        assert_eq!(
            ledger.succeed_request(request).unwrap_err(),
            Error::InvalidSessionState
        );
        ledger.mark_verified(request).expect("verified");
        ledger.succeed_request(request).expect("succeeded");
        assert_eq!(
            ledger.fail_request(request).unwrap_err(),
            Error::InvalidSessionState
        );
        assert_eq!(
            ledger.mark_proved(request).unwrap_err(),
            Error::InvalidSessionState
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_verification_marks_the_request_failed_and_blocks_retry() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request_id = request(137);
        let seed = root(138);
        ledger
            .begin_request(request_id, seed, root(139))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 140, 20, 100, 141, 142);
        ledger
            .reserve_oracle(request_id, oracle, stream_for(&oracle, 143), budget(1, 8))
            .expect("reserve");
        ledger.mark_proved(request_id).expect("proved");
        ledger
            .fail_request(request_id)
            .expect("failed verification");
        assert_eq!(
            ledger.mark_verified(request_id).unwrap_err(),
            Error::InvalidSessionState
        );
        // A retry of the same request or the same seed is rejected.
        let retry = request(144);
        ledger
            .begin_request(retry, root(145), root(146))
            .expect("retry request");
        assert_eq!(
            ledger
                .reserve_oracle(retry, oracle, stream_for(&oracle, 143), budget(1, 8),)
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        assert_eq!(
            ledger
                .begin_request(request(147), seed, root(148))
                .unwrap_err(),
            Error::StateRollbackSuspected
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_material_reservations_have_one_winner() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(150);
        ledger
            .begin_request(request, root(151), root(152))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 153, 20, 100, 154, 155);
        let stream = stream_for(&oracle, 156);
        let first = ledger.clone();
        let second = ledger.clone();
        let a = thread::spawn(move || {
            first
                .reserve_oracle(request, oracle, stream, budget(1, 8))
                .is_ok()
        });
        let b = thread::spawn(move || {
            second
                .reserve_oracle(request, oracle, stream, budget(1, 8))
                .is_ok()
        });
        let wins = [a.join().expect("join"), b.join().expect("join")]
            .into_iter()
            .filter(|won| *won)
            .count();
        assert_eq!(wins, 1);
        let state = ledger.read_state().expect("state");
        assert_eq!(state.oracle(oracle).expect("oracle").requests.len(), 1);
        assert_eq!(state.streams.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_seed_commitments_have_one_winner() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let seed = root(157);
        let first = ledger.clone();
        let second = ledger.clone();
        let a = thread::spawn(move || first.begin_request(request(158), seed, root(159)).is_ok());
        let b = thread::spawn(move || second.begin_request(request(160), seed, root(161)).is_ok());
        let wins = [a.join().expect("join"), b.join().expect("join")]
            .into_iter()
            .filter(|won| *won)
            .count();
        assert_eq!(wins, 1);
        assert_eq!(ledger.read_state().expect("state").requests.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_lock_fails_closed_for_the_opening_layer() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(162);
        ledger
            .begin_request(request, root(163), root(164))
            .expect("begin");
        std::fs::write(dir.join(LEDGER_LOCK), b"stale").expect("lock");
        let oracle = oracle(OracleKind::Weight, 165, 20, 100, 166, 167);
        assert_eq!(
            ledger
                .reserve_oracle(request, oracle, stream_for(&oracle, 168), budget(1, 8),)
                .unwrap_err(),
            Error::StateLocked
        );
        std::fs::remove_file(dir.join(LEDGER_LOCK)).expect("remove lock");
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 168), budget(1, 8))
            .expect("reserve after recovery");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_or_edited_event_log_is_rejected() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(170);
        ledger
            .begin_request(request, root(171), root(172))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 173, 20, 100, 174, 175);
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 176), budget(1, 8))
            .expect("reserve");
        ledger
            .record_openings(
                request,
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(1, 1, 1),
                &[9],
            )
            .expect("opening");
        let path = dir.join(LEDGER_FILE);
        let original = std::fs::read(&path).expect("read state");
        let restore = || std::fs::write(&path, &original).expect("restore state");

        let mut state: LedgerState = decode_bytes(&original).expect("decode");
        state.seq -= 1;
        std::fs::write(&path, encode_bytes(&state).expect("encode")).expect("write");
        assert!(ledger.read_state().is_err());
        restore();

        let mut state: LedgerState = decode_bytes(&original).expect("decode");
        state.events.pop();
        std::fs::write(&path, encode_bytes(&state).expect("encode")).expect("write");
        assert!(ledger.read_state().is_err());
        restore();

        let mut state: LedgerState = decode_bytes(&original).expect("decode");
        state.events[0].positions[0] = 10;
        std::fs::write(&path, encode_bytes(&state).expect("encode")).expect("write");
        assert!(ledger.read_state().is_err());
        restore();

        let mut state: LedgerState = decode_bytes(&original).expect("decode");
        state.head = Digest::ZERO;
        std::fs::write(&path, encode_bytes(&state).expect("encode")).expect("write");
        assert!(ledger.read_state().is_err());
        restore();

        ledger.read_state().expect("restored state");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn table_cumulative_union_is_checked_across_requests() {
        let dir = temp_dir();
        let policy = LedgerPolicy {
            table_reuse: TableReusePolicy::Budget {
                max_requests: 3,
                max_logical_openings: 100,
            },
        };
        let ledger = CommitmentLedger::initialize(&dir, policy).expect("init");
        let oracle = oracle(OracleKind::Table, 180, 0, 0, 181, 182);
        let stream = stream_for(&oracle, 183);
        let consume = |request_seed: u8, seed_seed: u8, positions: &[u64]| {
            let request = request(request_seed);
            ledger
                .begin_request(request, root(seed_seed), root(199))
                .expect("begin");
            ledger
                .reserve_oracle(request, oracle, stream, budget(positions.len() as u64, 4))
                .expect("table reserve");
            ledger
                .bind_commitment(request, oracle, root(184))
                .expect("table root");
            let tail = ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Table,
                    OpeningSource::TableColumn,
                    counts(
                        positions.len() as u64,
                        positions.len() as u64,
                        positions.len() as u64,
                    ),
                    positions,
                )
                .expect("table opening");
            ledger.mark_proved(request).expect("proved");
            ledger.mark_verified(request).expect("verified");
            ledger.succeed_request(request).expect("succeeded");
            tail
        };
        assert_eq!(consume(185, 186, &[1, 2]).new_union, 2);
        assert_eq!(consume(187, 188, &[3, 4]).new_union, 4);
        // The third request would push the distinct-position union to five on
        // a tree whose tail rank is four. It is rejected before the columns
        // are read, even though the logical total stays inside the policy.
        let request = request(189);
        ledger
            .begin_request(request, root(190), root(199))
            .expect("begin");
        ledger
            .reserve_oracle(request, oracle, stream, budget(1, 4))
            .expect("table reserve");
        assert_eq!(
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Table,
                    OpeningSource::TableColumn,
                    counts(1, 1, 1),
                    &[5],
                )
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let state = ledger.read_state().expect("state");
        let record = state.oracle(oracle).expect("oracle");
        assert_eq!(record.opened_positions, BTreeSet::from([1, 2, 3, 4]));
        assert_eq!(record.unique_positions, 4);
        assert_eq!(
            state.request(request).expect("request").state,
            RequestState::Failed
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Replays the exact opening-event table of the hidden output-head path.
    /// The table has one event per tree and stage; repeated logical draws are
    /// carried in the counts, never as duplicated positions. This replay fixes
    /// the pre-closure artifact's table duplicate case (603 draws, 601 distinct
    /// positions), so its unique total is 885. The challenge-dependent table
    /// union is measured by the formal fixture; the closed run measured 603
    /// distinct columns and a unique total of 887.
    #[test]
    fn the_hidden_output_head_event_table_reconciles() {
        let dir = temp_dir();
        let ledger = CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(210);
        ledger
            .begin_request(request, root(211), root(212))
            .expect("begin");

        let reserve = |oracle: OracleKey, material: u64, declared: u64, rank: u64| {
            ledger
                .reserve_oracle(
                    request,
                    oracle,
                    stream_for(&oracle, material),
                    budget(declared, rank),
                )
                .expect("reserve oracle");
        };
        let record = |oracle: OracleKey,
                      stage: Stage,
                      source: OpeningSource,
                      counts: OpeningCounts,
                      positions: Vec<u64>| {
            ledger
                .record_openings(request, oracle, stage, source, counts, &positions)
                .expect("record opening");
        };

        // Activation: 16 rows allocated at commit, never opened.
        let activation = oracle(OracleKind::Activation, 220, 0, 0xE001, 221, 222);
        ledger
            .reserve_oracle(
                request,
                activation,
                stream_for(&activation, 213),
                OracleBudget {
                    logical_draws: 0,
                    tail_rank: 0,
                    allocated_rows: 16,
                    committed_entries: 0,
                },
            )
            .expect("activation reservation");

        // Sixteen W oracles: two SEND draws and two own-BASE positions each.
        for index in 0..16u64 {
            let oracle = oracle(OracleKind::Weight, 230, 20 + index, 100 + index, 231, 232);
            reserve(oracle, 214, 4, 8192);
            record(
                oracle,
                Stage::Send,
                OpeningSource::SendSample,
                counts(2, 2, 2),
                range(index * 10, 2),
            );
            record(
                oracle,
                Stage::Base,
                OpeningSource::BaseMember,
                counts(2, 2, 2),
                range(index * 10 + 2, 2),
            );
        }

        // The committed table: 603 logical column draws, 601 distinct
        // positions and 601 response entries.
        let table = oracle(OracleKind::Table, 233, 0, 0, 234, 235);
        reserve(table, 215, 603, 8192);
        record(
            table,
            Stage::Table,
            OpeningSource::TableColumn,
            counts(603, 601, 601),
            range(0, 601),
        );

        // Index mask, shown map, Omega (shared and own), G_u and fresh masks.
        let index_mask = oracle(OracleKind::IndexMask, 236, 40, 202, 237, 238);
        reserve(index_mask, 216, 4, 256);
        record(
            index_mask,
            Stage::Send,
            OpeningSource::SendSample,
            counts(2, 2, 2),
            vec![1000, 1001],
        );
        record(
            index_mask,
            Stage::Base,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![1002, 1003],
        );
        let dot = oracle(OracleKind::DotMap, 239, 41, 302, 240, 241);
        reserve(dot, 217, 2, 8192);
        record(
            dot,
            Stage::Base,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![1100, 1101],
        );
        let omega = oracle(OracleKind::Omega, 242, 11, 301, 243, 244);
        reserve(omega, 218, 4, 8192);
        record(
            omega,
            Stage::Shared,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![1200, 1201],
        );
        record(
            omega,
            Stage::Base,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![1202, 1203],
        );
        let gu = oracle(OracleKind::Gu, 245, 10, 300, 246, 247);
        reserve(gu, 219, 2, 8192);
        record(
            gu,
            Stage::Shared,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![1300, 1301],
        );
        let fresh_shared = oracle(OracleKind::FreshMask, 248, 50, 200, 249, 250);
        reserve(fresh_shared, 251, 2, 256);
        record(
            fresh_shared,
            Stage::Shared,
            OpeningSource::BaseFresh,
            counts(2, 2, 2),
            vec![1400, 1401],
        );
        let fresh_own = oracle(OracleKind::FreshMask, 252, 1000, 201, 249, 250);
        reserve(fresh_own, 253, 2, 256);
        record(
            fresh_own,
            Stage::Base,
            OpeningSource::BaseFresh,
            counts(2, 2, 2),
            vec![1402, 1403],
        );
        for (kind, identity, oracle_id, material, stage, positions) in [
            (
                OracleKind::Star,
                254,
                0,
                260,
                Stage::Shared,
                vec![1500, 1501],
            ),
            (
                OracleKind::StarMask,
                255,
                1,
                261,
                Stage::Shared,
                vec![1502, 1503],
            ),
            (OracleKind::Star, 256, 0, 262, Stage::Base, vec![1504, 1505]),
            (
                OracleKind::StarMask,
                257,
                1,
                263,
                Stage::Base,
                vec![1506, 1507],
            ),
        ] {
            let oracle = oracle(kind, identity, oracle_id, 0, 258, 259);
            reserve(oracle, material, 2, 0);
            record(
                oracle,
                stage,
                OpeningSource::BaseStar,
                counts(2, 2, 2),
                positions,
            );
        }

        // The closed output relation. The slack, packed-bit and product-map
        // messages have one oracle per position, each opened at SEND and at
        // own BASE. The mask-only closure message adds its SEND and own-BASE
        // openings.
        for (kind, base, identity, material, start) in [
            (OracleKind::ClosureSlack, 300, 270, 280, 2000),
            (OracleKind::ClosureBits, 400, 271, 281, 3000),
            (OracleKind::ClosureMap, 500, 272, 282, 4000),
        ] {
            for index in 0..16u64 {
                let oracle = oracle(kind, identity, base + index, base + index, 283, 284);
                reserve(oracle, material, 4, 8192);
                record(
                    oracle,
                    Stage::Send,
                    OpeningSource::SendSample,
                    counts(2, 2, 2),
                    range(start + index * 10, 2),
                );
                record(
                    oracle,
                    Stage::Base,
                    OpeningSource::BaseMember,
                    counts(2, 2, 2),
                    range(start + index * 10 + 2, 2),
                );
            }
        }
        let closure_mask = oracle(OracleKind::ClosureMask, 285, 140, 500, 286, 287);
        reserve(closure_mask, 288, 4, 256);
        record(
            closure_mask,
            Stage::Send,
            OpeningSource::SendSample,
            counts(2, 2, 2),
            vec![5000, 5001],
        );
        record(
            closure_mask,
            Stage::Base,
            OpeningSource::BaseMember,
            counts(2, 2, 2),
            vec![5002, 5003],
        );
        let state = ledger.read_state().expect("state");
        assert_eq!(state.total_event_draws().expect("draws"), 887);
        assert_eq!(state.total_event_unique_positions().expect("unique"), 885);
        assert_eq!(state.events.len(), 143);
        assert_eq!(state.seq, 143);
        assert_eq!(
            state
                .totals_by_kind(OracleKind::Activation)
                .expect("activation"),
            OpeningCounts::default()
        );
        assert_eq!(
            state.totals_by_kind(OracleKind::Table).expect("table"),
            counts(603, 601, 601)
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::Weight)
                .expect("w")
                .logical_draws,
            64
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::IndexMask)
                .expect("mask")
                .logical_draws,
            4
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::DotMap)
                .expect("dot")
                .logical_draws,
            2
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::Omega)
                .expect("omega")
                .logical_draws,
            4
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::Gu)
                .expect("gu")
                .logical_draws,
            2
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::FreshMask)
                .expect("fresh")
                .logical_draws,
            4
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::Star)
                .expect("star")
                .unique_positions,
            4
        );
        assert_eq!(
            state
                .totals_by_kind(OracleKind::StarMask)
                .expect("star mask")
                .unique_positions,
            4
        );
        for kind in [
            OracleKind::ClosureSlack,
            OracleKind::ClosureBits,
            OracleKind::ClosureMap,
        ] {
            assert_eq!(
                state.totals_by_kind(kind).expect("closure").logical_draws,
                64
            );
        }
        assert_eq!(
            state
                .totals_by_kind(OracleKind::ClosureMask)
                .expect("closure mask")
                .logical_draws,
            4
        );
        assert_eq!(
            state.totals_by_kind(OracleKind::Switch).expect("switch"),
            OpeningCounts::default()
        );
        assert_eq!(
            state.totals_by_kind(OracleKind::Cs).expect("cs"),
            OpeningCounts::default()
        );
        // The pre-closure table is the closed table minus the closure events.
        let mut closure_draws = 0u64;
        for kind in [
            OracleKind::ClosureSlack,
            OracleKind::ClosureBits,
            OracleKind::ClosureMap,
            OracleKind::ClosureMask,
        ] {
            closure_draws += state.totals_by_kind(kind).expect("closure").logical_draws;
        }
        assert_eq!(closure_draws, 196);
        assert_eq!(
            state.total_event_draws().expect("draws") - closure_draws,
            691
        );
        assert_eq!(
            state.total_event_unique_positions().expect("unique") - closure_draws,
            689
        );

        ledger.mark_proved(request).expect("proved");
        ledger.mark_verified(request).expect("verified");
        ledger.succeed_request(request).expect("succeeded");
        let final_record = ledger
            .request_record(request)
            .expect("read")
            .expect("request");
        assert_eq!(final_record.state, RequestState::Succeeded);
        assert_eq!(final_record.event_count(), 143);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_declared_logical_overrun_fails_closed() {
        let dir = temp_dir();
        let ledger =
            CommitmentLedger::initialize(&dir, LedgerPolicy::default()).expect("init");
        let request = request(230);
        ledger
            .begin_request(request, root(231), root(232))
            .expect("begin");
        let oracle = oracle(OracleKind::Weight, 233, 20, 100, 234, 235);
        ledger
            .reserve_oracle(request, oracle, stream_for(&oracle, 236), budget(2, 8))
            .expect("reserve");
        for position in [1, 2] {
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Send,
                    OpeningSource::SendSample,
                    counts(1, 1, 1),
                    &[position],
                )
                .expect("opening inside the declared budget");
        }
        assert_eq!(
            ledger
                .record_openings(
                    request,
                    oracle,
                    Stage::Second,
                    OpeningSource::BaseMember,
                    counts(1, 1, 1),
                    &[3],
                )
                .unwrap_err(),
            Error::LeafBudgetExhausted
        );
        let state = ledger.read_state().expect("state");
        assert_eq!(
            state.request(request).expect("request").state,
            RequestState::Failed
        );
        let record = state.oracle(oracle).expect("oracle");
        assert_eq!(record.state, EventState::Aborted);
        assert_eq!(record.logical_openings, 2);
        assert_eq!(record.opened_positions, BTreeSet::from([1, 2]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
