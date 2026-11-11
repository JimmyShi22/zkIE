//! The authoritative cross-request commitment ledger.
//!
//! The ledger is a durable record keyed by commitment kind and identity. It
//! records the actual commitment once, every logical opening reservation,
//! table-root reuse under an explicit policy, and the state of each request or
//! attempt. A reservation is durable before proof work starts; the returned
//! claim has no release path, so dropping or forgetting it cannot make the
//! commitment reusable.
//!
//! Updates use an exclusive lock, a temporary file, a file sync, a rename and
//! a parent-directory sync. A crash before the rename leaves the old state
//! plus a lock or a temporary file, and every later reservation fails closed.
//! A crash after the rename leaves the spent state or a lock that also fails
//! closed. There is no rollback API. A filesystem rollback to an earlier
//! directory snapshot is an external deployment trust assumption and is not
//! detected here.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::lm_head_stage::{
    decode_bytes, encode_bytes, ensure_private_dir, private_options, read_bytes, sync_dir,
    write_bytes,
};

/// The ledger schema version. It covers the state file and every record.
pub const COMMITMENT_LEDGER_VERSION: u64 = 2;

const LEDGER_FILE: &str = "commitments.state";
const LEDGER_LOCK: &str = "commitments.lock";
const LEDGER_TEMP: &str = "commitments.tmp";

/// How long a reservation waits for a concurrent writer before failing closed.
const LOCK_ATTEMPTS: u32 = 100;
const LOCK_RETRY: Duration = Duration::from_millis(2);

/// The kind of commitment a ledger entry names. A non-table commitment may be
/// reserved once and is then spent; a table commitment may be reused only when
/// the ledger policy explicitly budgets it.
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

/// The state of one request or attempt against one commitment entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptState {
    /// The reservation is durable and the proof work has not finished.
    Reserved,
    /// The proof work finished and the caller recorded success.
    Succeeded,
    /// The proof work finished and the caller recorded failure.
    Failed,
}

/// One durable request or attempt record.
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

/// The complete durable ledger state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerState {
    pub version: u64,
    pub policy: LedgerPolicy,
    pub commitments: BTreeMap<LedgerKey, CommitmentRecord>,
}

impl LedgerState {
    pub fn record(&self, key: LedgerKey) -> Option<&CommitmentRecord> {
        self.commitments.get(&key)
    }

    /// The total logical openings reserved by every commitment.
    pub fn total_logical_openings(&self) -> Result<u64> {
        self.commitments.values().try_fold(0u64, |total, record| {
            total
                .checked_add(record.logical_openings)
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
        Ok(())
    }
}

/// One durable reservation. The reservation is already committed when this
/// value is returned; dropping it leaves the attempt in [`AttemptState::Reserved`]
/// and does not release anything. Call [`LedgerClaim::succeed`] or
/// [`LedgerClaim::fail`] to record the outcome of the proof work.
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
            commitments: BTreeMap::new(),
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

    /// Reserves one commitment for one request or attempt before proof work.
    /// The reservation records the logical opening count, counts table reuse
    /// under the stored policy and returns a durable claim.
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
            CommitmentKind::Activation | CommitmentKind::Weight | CommitmentKind::Mask
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

    /// Reads and validates the authoritative state. A concurrent writer's
    /// temporary file is ignored; the state file is replaced atomically.
    pub fn read_state(&self) -> Result<LedgerState> {
        let state: LedgerState = decode_bytes(&read_bytes(&self.dir.join(LEDGER_FILE))?)?;
        state.validate()?;
        Ok(state)
    }

    /// Reads one commitment entry from a validated snapshot.
    pub fn record_for(&self, kind: CommitmentKind, identity: Digest) -> Result<Option<CommitmentRecord>> {
        let key = LedgerKey::new(kind, identity)?;
        Ok(self.read_state()?.record(key).cloned())
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

fn finish_attempt(
    dir: &Path,
    key: LedgerKey,
    request: Digest,
    state: AttemptState,
) -> Result<()> {
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
        assert_eq!(record.attempt(attempt).expect("attempt").state, AttemptState::Reserved);
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
        let identity = digest_from_hex(
            &std::env::var("ZKIE_LEDGER_WORKER_IDENTITY").expect("identity"),
        );
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
        let status = spawn_worker(
            &dir,
            CommitmentKind::Weight,
            identity,
            attempt,
            4,
            "none",
        )
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
}
