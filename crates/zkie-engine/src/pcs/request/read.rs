//! Column reads: drawn positions, a read budget, a verifier value store and
//! a prover opening cache (foundation only).
//!
//! The verifier draws positions from its own chain; the prover gets the same
//! positions from the same transcript. The read budget counts supplied column
//! reads, not logical draws: a low-level entry charges the list it receives,
//! while `FoldeqSession::read_queries` verifies unique physical positions and
//! charges that deduplicated list. The plan's logical draw count is a separate
//! quantity. The provider's hiding budget is a separate ledger; verification
//! deduplication does not reduce it, and no cumulative hiding-budget closure is
//! claimed. A position read twice must return the same column values, and an
//! exhausted budget fails closed. This module is not wired into production;
//! the plan/priced_table that fixes the budget is still open.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::chain::Chain;
use crate::pcs::request::field::Fp;
use crate::pcs::request::table::{
    verify_columns, ColumnOpening, IndexReader,
};
#[cfg(test)]
use crate::pcs::request::table::TestTable;

/// Draw `count` positions below `2^log_n` from the chain. The order is the
/// draw order; a caller that wants one opening list sorts and deduplicates
/// with `unique_sorted`.
pub fn draw_positions(chain: &mut Chain, count: usize, log_n: u32) -> Result<Vec<u64>> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        positions.push(chain.draw_position(log_n)?);
    }
    Ok(positions)
}

/// The drawn positions in increasing order without repeats. The number of
/// draws is not the number of distinct positions. A low-level read entry
/// charges the list it receives; a caller that treats a physical opening as
/// the budget unit must deduplicate first. The plan's logical draw count stays
/// a separate quantity.
pub fn unique_sorted(positions: &[u64]) -> Vec<u64> {
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

/// A read budget in supplied column reads. One charge is one entry of the
/// list an entry verifies; it is not necessarily one logical draw. Charging
/// past the maximum fails closed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(test)]
pub struct ReadBudget {
    max_reads: u64,
    used: u64,
}

#[cfg(test)]
impl ReadBudget {
    pub fn new(max_reads: u64) -> Self {
        Self { max_reads, used: 0 }
    }

    pub fn charge(&mut self, reads: usize) -> Result<()> {
        let reads = u64::try_from(reads).map_err(|_| Error::ResourceLimit)?;
        let used = self.used.checked_add(reads).ok_or(Error::ResourceLimit)?;
        if used > self.max_reads {
            return Err(Error::ResourceLimit);
        }
        self.used = used;
        Ok(())
    }

    /// Return a local read-budget charge that never led to an opening because
    /// the shared hiding token rejected the same request first. This is not a
    /// refund of the provider's long-term hiding reservation; that reservation
    /// is never returned. A charge made after the token accepted the request is
    /// not refunded here.
    fn refund_unstarted(&mut self, reads: usize) -> Result<()> {
        let reads = u64::try_from(reads).map_err(|_| Error::ResourceLimit)?;
        self.used = self
            .used
            .checked_sub(reads)
            .ok_or(Error::InvalidSessionState)?;
        Ok(())
    }

    pub fn max_reads(&self) -> u64 {
        self.max_reads
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn remaining(&self) -> u64 {
        self.max_reads.saturating_sub(self.used)
    }
}

/// A reservation made against a read ledger. A token is committed once or
/// cancelled once; a forged or replayed token fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadReservation {
    serial: u64,
    reads: u64,
}

/// The verifier's cross-batch read ledger. It lives with the statement, not
/// with one session, so starting a new session cannot reset the budget. Reads
/// are reserved atomically before verification and committed after success;
/// a failed read cancels its reservation, while a crash leaves it outstanding
/// and `recover` counts it as used (fail closed). The unit is one supplied
/// column read (one list entry); a caller that budgets physical openings must
/// deduplicate before reserving. The plan's logical draw count and the
/// provider's hiding ledger are separate quantities.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadLedger {
    max_reads: u64,
    used: u64,
    reserved: u64,
    next_serial: u64,
    outstanding: BTreeMap<u64, u64>,
}

impl ReadLedger {
    pub fn new(max_reads: u64) -> Self {
        Self {
            max_reads,
            used: 0,
            reserved: 0,
            next_serial: 0,
            outstanding: BTreeMap::new(),
        }
    }

    pub fn max_reads(&self) -> u64 {
        self.max_reads
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding.len()
    }

    /// Reserve `reads` atomically. Exhaustion fails before any verification.
    pub fn reserve(&mut self, reads: usize) -> Result<ReadReservation> {
        let reads = u64::try_from(reads).map_err(|_| Error::ResourceLimit)?;
        let used = self
            .used
            .checked_add(self.reserved)
            .and_then(|total| total.checked_add(reads))
            .ok_or(Error::ResourceLimit)?;
        if used > self.max_reads {
            return Err(Error::ResourceLimit);
        }
        let serial = self.next_serial;
        self.next_serial = self.next_serial.checked_add(1).ok_or(Error::ResourceLimit)?;
        self.outstanding.insert(serial, reads);
        self.reserved = self
            .reserved
            .checked_add(reads)
            .ok_or(Error::ResourceLimit)?;
        Ok(ReadReservation { serial, reads })
    }

    /// Commit a successful read. A missing or mismatched token fails.
    pub fn commit(&mut self, reservation: ReadReservation) -> Result<()> {
        let reads = self
            .outstanding
            .remove(&reservation.serial)
            .ok_or(Error::StatementMismatch)?;
        if reads != reservation.reads {
            return Err(Error::StatementMismatch);
        }
        self.reserved = self
            .reserved
            .checked_sub(reads)
            .ok_or(Error::StatementMismatch)?;
        self.used = self
            .used
            .checked_add(reads)
            .ok_or(Error::ResourceLimit)?;
        Ok(())
    }

    /// Cancel a read that failed before its result was accepted.
    pub fn cancel(&mut self, reservation: ReadReservation) -> Result<()> {
        let reads = self
            .outstanding
            .remove(&reservation.serial)
            .ok_or(Error::StatementMismatch)?;
        if reads != reservation.reads {
            return Err(Error::StatementMismatch);
        }
        self.reserved = self
            .reserved
            .checked_sub(reads)
            .ok_or(Error::StatementMismatch)?;
        Ok(())
    }

    /// Recovery: an outstanding reservation may have been read before a crash,
    /// so it is counted as used and cleared. This can only spend budget, never
    /// return it.
    pub fn recover(&mut self) {
        self.used = self.used.saturating_add(self.reserved);
        self.reserved = 0;
        self.outstanding.clear();
    }
}

/// The provider's table-life hiding state. The unit is one column. It is keyed
/// by the table and a lifecycle number, so the same state cannot be applied to
/// another leaf/table or a new lifecycle. `begin` charges before any fold or
/// send and never refunds: an abandoned proof has spent its columns all the
/// same. The provider keeps the state authoritative and may save it atomically;
/// cross-process locking, rollback protection and deriving the maximum from the
/// encoded random tail and opening budgets are not implemented, and an
/// independent verifier without trusted global state does not claim to verify
/// the cumulative hidden budget. This is separate from the read ledger.
/// The token a provider receives after a hiding pre-charge for one proof.
/// Its fields are private and only `HidingState::begin_charge` produces it.
/// Copies share one consumed-position set, so copying the token or using it
/// from several readers does not create new budget: the union of physical
/// positions never exceeds the proof's reserved total, while a repeat of an
/// already opened position is free. The charge is in-memory; the durable
/// provider ledger is open.
#[derive(Clone, Debug)]
pub struct HidingCharge {
    table: Digest,
    lifecycle: u64,
    total: usize,
    state: Arc<Mutex<ChargeState>>,
}

#[derive(Debug, Default)]
struct ChargeState {
    unique: HashSet<u64>,
    units: usize,
}

impl HidingCharge {
    pub fn table(&self) -> Digest {
        self.table
    }

    pub fn lifecycle(&self) -> u64 {
        self.lifecycle
    }

    pub fn columns(&self) -> usize {
        self.total
    }

    /// Consume the new physical positions and return how many were new to the
    /// shared token. A repeat of a consumed position adds nothing; the union
    /// above the reserved total fails. The returned count is token-global and
    /// is not a reader's read budget: a second reader must still budget the
    /// positions it opens even when this token reports zero fresh entries.
    pub fn consume(&self, positions: &[u64]) -> Result<usize> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        // Build the candidate set and check the total before committing
        // anything. A failed request leaves the consumed set unchanged, so a
        // retry cannot slip a new position in through an earlier partial
        // insertion.
        let mut fresh = HashSet::new();
        for position in positions {
            if !state.unique.contains(position) {
                fresh.insert(*position);
            }
        }
        let candidate = state
            .units
            .checked_add(fresh.len())
            .ok_or(Error::ResourceLimit)?;
        if candidate > self.total {
            return Err(Error::ResourceLimit);
        }
        for position in &fresh {
            state.unique.insert(*position);
        }
        state.units = candidate;
        Ok(fresh.len())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HidingState {
    table: Digest,
    lifecycle: u64,
    max_columns: u64,
    used: u64,
}

impl HidingState {
    pub fn new(table: Digest, lifecycle: u64, max_columns: u64) -> Self {
        Self {
            table,
            lifecycle,
            max_columns,
            used: 0,
        }
    }

    pub fn table(&self) -> Digest {
        self.table
    }

    pub fn lifecycle(&self) -> u64 {
        self.lifecycle
    }

    pub fn max_columns(&self) -> u64 {
        self.max_columns
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    fn begin(&mut self, table: Digest, lifecycle: u64, columns: usize) -> Result<()> {
        if table != self.table || lifecycle != self.lifecycle {
            return Err(Error::StatementMismatch);
        }
        let columns = u64::try_from(columns).map_err(|_| Error::ResourceLimit)?;
        let used = self.used.checked_add(columns).ok_or(Error::ResourceLimit)?;
        if used > self.max_columns {
            return Err(Error::ResourceLimit);
        }
        self.used = used;
        Ok(())
    }

    /// Charge the table-life hiding capacity and return the token that the
    /// charged read/open path requires. The charge is an in-memory update, not
    /// a durable pre-charge; the durable provider ledger is still open.
    pub fn begin_charge(
        &mut self,
        table: Digest,
        lifecycle: u64,
        columns: usize,
    ) -> Result<HidingCharge> {
        self.begin(table, lifecycle, columns)?;
        Ok(HidingCharge {
            table,
            lifecycle,
            total: columns,
            state: Arc::new(Mutex::new(ChargeState::default())),
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut writer = crate::primitives::codec::CanonicalWriter::new();
        writer.put_digest(&self.table);
        writer.put_u64(self.lifecycle);
        writer.put_u64(self.max_columns);
        writer.put_u64(self.used);
        writer.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader =
            crate::primitives::codec::CanonicalReader::new(bytes, zkie_core::common::codec::DecodeLimits::unlimited());
        let table = reader.get_digest()?;
        let lifecycle = reader.get_u64()?;
        let max_columns = reader.get_u64()?;
        let used = reader.get_u64()?;
        reader.finish()?;
        if used > max_columns {
            return Err(Error::InvalidEncoding);
        }
        Ok(Self {
            table,
            lifecycle,
            max_columns,
            used,
        })
    }

    /// Write the state to `path` through a temporary file and an atomic rename.
    /// This gives a durable single-writer update; it does not lock across
    /// processes, prevent a rollback to an older valid file, or anchor the
    /// state outside the provider.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let temporary = path.with_extension("tmp");
        let mut file = std::fs::File::create(&temporary).map_err(|_| Error::Io)?;
        file.write_all(&self.encode()).map_err(|_| Error::Io)?;
        file.sync_all().map_err(|_| Error::Io)?;
        std::fs::rename(&temporary, path).map_err(|_| Error::Io)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|_| Error::Io)?;
        Self::decode(&bytes)
    }
}

/// The verifier's column reads: a shared ledger and one value store per
/// session. The ledger is the authority for the budget; the value store keeps
/// repeated reads consistent.
#[derive(Clone, Debug)]
pub struct VerifierReads {
    ledger: Arc<Mutex<ReadLedger>>,
    values: HashMap<(Digest, u64), Vec<Fp>>,
}

impl VerifierReads {
    /// The ledger must come from the statement; a session cannot reset it.
    pub fn new(ledger: Arc<Mutex<ReadLedger>>) -> Self {
        Self {
            ledger,
            values: HashMap::new(),
        }
    }

    pub fn ledger(&self) -> Arc<Mutex<ReadLedger>> {
        Arc::clone(&self.ledger)
    }

    /// Verify the openings for `positions`, charge one read per supplied
    /// list entry and remember each column. This entry does not deduplicate:
    /// callers that budget physical openings must pass a unique list. The
    /// positions come from the verifier's chain or its plan, never from the
    /// proof; a repeated position must carry the same column values.
    pub fn read(
        &mut self,
        reader: &IndexReader,
        positions: &[u64],
        openings: &[ColumnOpening],
    ) -> Result<Vec<Vec<Fp>>> {
        if positions.len() != openings.len() {
            return Err(Error::ShapeMismatch);
        }
        let reservation = self
            .ledger
            .lock()
            .map_err(|_| Error::InvalidSessionState)?
            .reserve(positions.len())?;
        let result = self.verify_and_remember(reader, positions, openings);
        match result {
            Ok(columns) => {
                self.ledger
                    .lock()
                    .map_err(|_| Error::InvalidSessionState)?
                    .commit(reservation)?;
                Ok(columns)
            }
            Err(error) => {
                if let Ok(mut ledger) = self.ledger.lock() {
                    let _ = ledger.cancel(reservation);
                }
                Err(error)
            }
        }
    }

    fn verify_and_remember(
        &mut self,
        reader: &IndexReader,
        positions: &[u64],
        openings: &[ColumnOpening],
    ) -> Result<Vec<Vec<Fp>>> {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(positions.len())
            .map_err(|_| Error::ResourceLimit)?;
        for (position, opening) in positions.iter().zip(openings) {
            if opening.position != *position {
                return Err(Error::InvalidIndex);
            }
            let values = verify_columns(
                reader,
                std::slice::from_ref(position),
                std::slice::from_ref(opening),
            )?;
            self.remember(reader.root, *position, &values)?;
            columns.push(values);
        }
        Ok(columns)
    }

    /// The repeated-read rule: the first value for a (root, position) is kept;
    /// a later read must return exactly the same bytes.
    fn remember(&mut self, root: Digest, position: u64, values: &[Fp]) -> Result<()> {
        match self.values.get(&(root, position)) {
            Some(known) if known.as_slice() == values => Ok(()),
            Some(_) => Err(Error::StatementMismatch),
            None => {
                self.values.insert((root, position), values.to_vec());
                Ok(())
            }
        }
    }

    pub fn values(&self, root: Digest, position: u64) -> Option<&[Fp]> {
        self.values.get(&(root, position)).map(Vec::as_slice)
    }

}

/// The prover's column reads: a budget and an opening cache. A repeated
/// position returns the opening first produced for it. Cloning copies the
/// budget and cache: a clone is a separate logical reader, not a share of one
/// global read budget, and this type does not claim a cross-instance budget.
#[derive(Clone, Debug, Default)]
#[cfg(test)]
pub struct ProverReads {
    budget: ReadBudget,
    openings: HashMap<(Digest, u64), ColumnOpening>,
}

#[cfg(test)]
impl ProverReads {
    pub fn new(max_reads: u64) -> Self {
        Self {
            budget: ReadBudget::new(max_reads),
            openings: HashMap::new(),
        }
    }

    /// Open the requested positions under one proof's hiding token. The read
    /// budget is per reader and is charged for every position this reader will
    /// open and has not already cached, before the shared token is touched. A
    /// budget failure leaves the token untouched. If the shared token rejects
    /// the request, the unstarted local charge is returned; the provider's
    /// long-term hiding reservation is never returned. Once the token accepts,
    /// neither it nor the local read charge is refunded, even if an opening
    /// later fails; a retry pays again for positions not already cached.
    pub fn open(
        &mut self,
        charge: &HidingCharge,
        table: &TestTable,
        positions: &[u64],
    ) -> Result<Vec<ColumnOpening>> {
        if charge.table() != table.reader.root {
            return Err(Error::StatementMismatch);
        }
        // The reader's budget is its own. It charges the positions this reader
        // will actually open, before the shared hiding token is touched; a
        // token-global zero-fresh result does not exempt a second reader. The
        // budget charge is local and may be returned only when the shared token
        // rejects the request before any opening; the provider's long-term
        // hiding reservation is never returned.
        let mut fresh_positions = Vec::new();
        fresh_positions
            .try_reserve_exact(positions.len())
            .map_err(|_| Error::ResourceLimit)?;
        let mut seen = HashSet::new();
        for position in positions {
            let key = (table.reader.root, *position);
            if self.openings.contains_key(&key) || !seen.insert(*position) {
                continue;
            }
            fresh_positions.push(*position);
        }
        self.budget.charge(fresh_positions.len())?;
        if let Err(error) = charge.consume(positions) {
            self.budget.refund_unstarted(fresh_positions.len())?;
            return Err(error);
        }
        let mut openings = Vec::new();
        openings
            .try_reserve_exact(positions.len())
            .map_err(|_| Error::ResourceLimit)?;
        for position in positions {
            let key = (table.reader.root, *position);
            if let Some(known) = self.openings.get(&key) {
                openings.push(known.clone());
                continue;
            }
            let mut fresh = table.open_columns_charged(charge, std::slice::from_ref(position))?;
            let opening = fresh.pop().ok_or(Error::InvalidProof)?;
            self.openings.insert(key, opening.clone());
            openings.push(opening);
        }
        Ok(openings)
    }

    pub fn budget(&self) -> ReadBudget {
        self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::chain::KIND_POSITIONS;
    use crate::pcs::request::code::build_code_plan;
    use crate::pcs::request::params::LEAF_SALT_MIN_BYTES;
    use crate::pcs::request::table::{test_commit_table, IndexShape, TableDistances};
    use std::sync::{Arc, Mutex};
    use zkie_core::common::field::PrimeCharacteristicRing;

    fn shape() -> IndexShape {
        IndexShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            v: 2,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn committed() -> TestTable {
        let shape = shape();
        let rows = shape.rows() as usize;
        let messages: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.ell)
                    .map(|index| Fp::from_u64((row * 10 + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let tails: Vec<Vec<Fp>> = (0..rows)
            .map(|row| {
                (0..shape.code.r)
                    .map(|index| Fp::from_u64((row + index as usize + 1) as u64))
                    .collect()
            })
            .collect();
        let salts: Vec<Vec<u8>> = (0..shape.code.n() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); shape.salt_bytes])
            .collect();
        test_commit_table(shape, Digest([7; 32]), &messages, &tails, &salts).unwrap()
    }

    fn chain() -> Chain {
        let mut chain = Chain::begin(7, &Digest::ZERO, b"positions", &[], None).unwrap();
        let prefix = chain.message().prefix(KIND_POSITIONS, 0).unwrap();
        chain.step(&prefix, 0).unwrap();
        chain
    }

    fn ledger(max_reads: u64) -> Arc<Mutex<ReadLedger>> {
        Arc::new(Mutex::new(ReadLedger::new(max_reads)))
    }

    fn used(reads: &VerifierReads) -> u64 {
        reads.ledger().lock().unwrap().used()
    }

    #[test]
    fn positions_are_drawn_in_range_and_deterministically() {
        let shape = shape();
        let mut first = chain();
        let mut second = chain();
        let left = draw_positions(&mut first, 8, shape.code.log_n).unwrap();
        let right = draw_positions(&mut second, 8, shape.code.log_n).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.len(), 8);
        assert!(left.iter().all(|position| *position < shape.code.n()));
        let unique = unique_sorted(&left);
        assert!(unique.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(unique.len() <= left.len());
        let mut empty = chain();
        assert!(draw_positions(&mut empty, 0, shape.code.log_n).unwrap().is_empty());
    }

    #[test]
    fn verifier_reads_charge_and_repeat_consistently() {
        let table = committed();
        let positions = [1u64, 4];
        let openings = table.open_columns(&positions).unwrap();
        let mut reads = VerifierReads::new(ledger(2));
        let columns = reads.read(&table.reader, &positions, &openings).unwrap();
        assert_eq!(columns.len(), 2);
        assert_eq!(used(&reads), 2);
        assert_eq!(reads.values(table.reader.root, 1), Some(columns[0].as_slice()));
        // A repeated read returns the remembered values and still charges.
        let again = reads.read(&table.reader, &positions[..1], &openings[..1]).unwrap_err();
        assert!(matches!(again, Error::ResourceLimit));
        let mut reads = VerifierReads::new(ledger(4));
        let first = reads.read(&table.reader, &positions, &openings).unwrap();
        let second = reads.read(&table.reader, &positions, &openings).unwrap();
        assert_eq!(first, second);
        assert_eq!(used(&reads), 4);
        // The value store rejects a conflicting column for the same address.
        let different = vec![Fp::from_u64(999); shape().rows() as usize];
        assert!(reads
            .remember(table.reader.root, positions[0], &different)
            .is_err());
        // A different root is a different address.
        assert!(reads
            .remember(Digest([8; 32]), positions[0], &different)
            .is_ok());
    }

    #[test]
    fn prover_reads_cache_and_bound_the_budget() {
        let table = committed();
        let positions = [1u64, 4];
        let root = table.reader.root;
        let mut state = HidingState::new(root, 1, 8);
        // The proof reserves two columns: the union of physical opens must not
        // exceed them, while repeats are free.
        let charge = state.begin_charge(root, 1, 2).unwrap();
        let mut reads = ProverReads::new(8);
        let first = reads.open(&charge, &table, &positions).unwrap();
        assert_eq!(reads.budget().used(), 2);
        let repeated = reads.open(&charge, &table, &positions).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(reads.budget().used(), 2);
        // A token copy and a second reader share the consumed set, but the
        // second reader still pays its own read budget for the opening it
        // makes, even when the token reports no fresh position.
        let copied = charge.clone();
        let mut second = ProverReads::new(8);
        assert!(second.open(&copied, &table, &[6]).is_err());
        assert!(second.open(&copied, &table, &[1]).is_ok());
        assert_eq!(second.budget().used(), 1);
        let other = Digest([8; 32]);
        let mut other_state = HidingState::new(other, 1, 8);
        let other_charge = other_state.begin_charge(other, 1, 8).unwrap();
        assert!(reads.open(&other_charge, &table, &positions).is_err());
    }

    #[test]
    fn prover_budget_failure_does_not_consume_the_hiding_token() {
        let table = committed();
        let root = table.reader.root;
        let mut state = HidingState::new(root, 1, 2);
        let charge = state.begin_charge(root, 1, 2).unwrap();
        let mut reads = ProverReads::new(1);
        assert!(reads.open(&charge, &table, &[1, 4]).is_err());
        assert_eq!(reads.budget().used(), 0);
        assert_eq!(reads.budget().remaining(), 1);
        // The retry still fails the reader's budget rather than seeing a
        // zero-fresh token and slipping through.
        assert!(reads.open(&charge, &table, &[1, 4]).is_err());
        assert_eq!(reads.budget().used(), 0);
        // No hiding position was consumed by the failed budget attempt, so
        // another request can still use the reserved two positions.
        assert_eq!(charge.consume(&[5, 6]).unwrap(), 2);
    }

    #[test]
    fn token_clone_does_not_exempt_a_second_reader_budget() {
        let table = committed();
        let root = table.reader.root;
        let mut state = HidingState::new(root, 1, 1);
        let charge = state.begin_charge(root, 1, 1).unwrap();
        let mut first = ProverReads::new(1);
        first.open(&charge, &table, &[0]).unwrap();
        // The token now sees position 0 as fresh=0; a reader with no budget
        // must still fail, and one with a budget pays for its own opening.
        let mut no_budget = ProverReads::new(0);
        assert!(no_budget.open(&charge, &table, &[0]).is_err());
        assert_eq!(no_budget.budget().used(), 0);
        let mut second = ProverReads::new(1);
        second.open(&charge, &table, &[0]).unwrap();
        assert_eq!(second.budget().used(), 1);
        let copied = charge.clone();
        let mut third = ProverReads::new(0);
        assert!(third.open(&copied, &table, &[0]).is_err());
        let mut fourth = ProverReads::new(1);
        fourth.open(&copied, &table, &[0]).unwrap();
        assert_eq!(fourth.budget().used(), 1);
    }

    #[test]
    fn concurrent_readers_charge_independent_budgets() {
        let table = Arc::new(committed());
        let root = table.reader.root;
        let mut state = HidingState::new(root, 1, 1);
        let charge = state.begin_charge(root, 1, 1).unwrap();
        let (left, right) = std::thread::scope(|scope| {
            let left_table = Arc::clone(&table);
            let left_charge = charge.clone();
            let left = scope.spawn(move || {
                let mut reads = ProverReads::new(1);
                let ok = reads.open(&left_charge, &left_table, &[0]).is_ok();
                (ok, reads.budget().used())
            });
            let right_table = Arc::clone(&table);
            let right_charge = charge.clone();
            let right = scope.spawn(move || {
                let mut reads = ProverReads::new(1);
                let ok = reads.open(&right_charge, &right_table, &[0]).is_ok();
                (ok, reads.budget().used())
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        // Both readers open the same already-consumed position: the shared
        // token says fresh=0 for one of them, but both readers pay their own
        // one-read budget.
        assert!(left.0 && right.0);
        assert_eq!(left.1 + right.1, 2);
    }

    #[test]
    fn concurrent_readers_refund_only_unstarted_budget() {
        let table = Arc::new(committed());
        let root = table.reader.root;
        let mut state = HidingState::new(root, 1, 1);
        let charge = state.begin_charge(root, 1, 1).unwrap();
        let (left, right) = std::thread::scope(|scope| {
            let left_table = Arc::clone(&table);
            let left_charge = charge.clone();
            let left = scope.spawn(move || {
                let mut reads = ProverReads::new(1);
                let ok = reads.open(&left_charge, &left_table, &[0]).is_ok();
                (ok, reads.budget().used())
            });
            let right_table = Arc::clone(&table);
            let right_charge = charge.clone();
            let right = scope.spawn(move || {
                let mut reads = ProverReads::new(1);
                let ok = reads.open(&right_charge, &right_table, &[1]).is_ok();
                (ok, reads.budget().used())
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        // One token position: exactly one reader opens and pays; the other
        // token rejection returns its unstarted local budget.
        assert_eq!(left.0 as u8 + right.0 as u8, 1);
        assert_eq!(left.1 + right.1, 1);
    }

    #[test]
    fn ledger_reservations_are_atomic_and_shared() {
        let shared = ledger(2);
        let mut first = VerifierReads::new(Arc::clone(&shared));
        let table = committed();
        let positions = [1u64];
        let openings = table.open_columns(&positions).unwrap();
        first.read(&table.reader, &positions, &openings).unwrap();
        assert_eq!(used(&first), 1);
        // The ledger is shared: a new session sees the used read and spends
        // the remaining budget rather than starting over.
        let mut second = VerifierReads::new(Arc::clone(&shared));
        second.read(&table.reader, &positions, &openings).unwrap();
        assert_eq!(shared.lock().unwrap().used(), 2);
        let mut third = VerifierReads::new(Arc::clone(&shared));
        assert!(third.read(&table.reader, &positions, &openings).is_err());
        // Reservations are atomic and never overspend.
        let mut ledger = ReadLedger::new(2);
        let reservation = ledger.reserve(1).unwrap();
        assert!(ledger.reserve(2).is_err());
        ledger.commit(reservation).unwrap();
        assert!(ledger.reserve(2).is_err());
        ledger.reserve(1).unwrap();
    }

    #[test]
    fn recovery_counts_outstanding_reservations_as_used() {
        let mut ledger = ReadLedger::new(3);
        let _reservation = ledger.reserve(2).unwrap();
        ledger.recover();
        assert_eq!(ledger.used(), 2);
        assert_eq!(ledger.reserved(), 0);
        assert_eq!(ledger.outstanding(), 0);
    }

    #[test]
    fn failed_read_cancels_its_reservation() {
        let table = committed();
        let positions = [1u64, 4];
        let openings = table.open_columns(&positions).unwrap();
        let shared = ledger(2);
        let mut reads = VerifierReads::new(Arc::clone(&shared));
        let mut tampered = openings.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(reads.read(&table.reader, &positions, &tampered).is_err());
        assert_eq!(shared.lock().unwrap().used(), 0);
        assert_eq!(shared.lock().unwrap().reserved(), 0);
        // The budget is still available after the failed read.
        reads.read(&table.reader, &positions, &openings).unwrap();
        assert_eq!(shared.lock().unwrap().used(), 2);
    }

    #[test]
    fn hiding_charge_failure_is_atomic() {
        let table = Digest([7; 32]);
        let mut state = HidingState::new(table, 1, 1);
        let charge = state.begin_charge(table, 1, 1).unwrap();
        assert_eq!(charge.consume(&[0]).unwrap(), 1);
        // The failed request consumes nothing; retrying it still fails.
        assert!(charge.consume(&[1]).is_err());
        assert!(charge.consume(&[1]).is_err());
        assert!(charge.consume(&[0, 1]).is_err());
        // An already consumed position remains free after the failures.
        assert_eq!(charge.consume(&[0]).unwrap(), 0);
        // A clone shares the same consumed set.
        let copied = charge.clone();
        assert_eq!(copied.consume(&[0]).unwrap(), 0);
        assert!(copied.consume(&[1]).is_err());
        assert_eq!(charge.columns(), 1);
    }

    #[test]
    fn over_limit_retry_stays_rejected_through_charged_opens() {
        let table = committed();
        let mut state = HidingState::new(table.reader.root, 1, 1);
        let charge = state.begin_charge(table.reader.root, 1, 1).unwrap();
        let third_charge = charge.clone();
        let mut first = ProverReads::new(2);
        let mut second = ProverReads::new(2);
        let mut third = ProverReads::new(2);
        first.open(&charge, &table, &[0]).unwrap();
        // A second reader cannot spend a new column and keeps failing on the
        // same over-limit position through the real charged-open path.
        assert!(second.open(&charge, &table, &[1]).is_err());
        assert!(second.open(&charge, &table, &[1]).is_err());
        assert!(second.open(&charge, &table, &[0, 1]).is_err());
        assert_eq!(second.open(&charge, &table, &[0]).unwrap().len(), 1);
        // A cloned token and a third reader share the same rejected state.
        assert!(third.open(&third_charge, &table, &[1]).is_err());
        assert_eq!(third.open(&third_charge, &table, &[0]).unwrap().len(), 1);
        // The table's charged open rejects the same position on retry and
        // still serves the already consumed one.
        assert!(table.open_columns_charged(&charge, &[1]).is_err());
        assert!(table.open_columns_charged(&charge, &[1]).is_err());
        assert!(table.open_columns_charged(&charge, &[0]).is_ok());
    }

    #[test]
    fn hiding_state_charges_before_send_and_never_refunds() {
        let table = Digest([7; 32]);
        let mut state = HidingState::new(table, 1, 3);
        state.begin_charge(table, 1, 2).unwrap();
        assert_eq!(state.used(), 2);
        assert!(state.begin_charge(table, 1, 2).is_err());
        state.begin_charge(table, 1, 1).unwrap();
        assert_eq!(state.used(), 3);
        state.begin_charge(table, 1, 0).unwrap();
        assert!(state.begin_charge(table, 1, 1).is_err());
        assert!(state.begin_charge(Digest([8; 32]), 1, 1).is_err());
        assert!(state.begin_charge(table, 2, 1).is_err());
    }

    #[test]
    fn hiding_state_saves_and_loads_atomically() {
        let table = Digest([7; 32]);
        let mut state = HidingState::new(table, 3, 4);
        state.begin_charge(table, 3, 2).unwrap();
        let path = std::env::temp_dir().join(format!(
            "zkie-hiding-state-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        state.save(&path).unwrap();
        let mut loaded = HidingState::load(&path).unwrap();
        assert_eq!(loaded, state);
        loaded.begin_charge(table, 3, 2).unwrap();
        assert!(loaded.begin_charge(table, 3, 2).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hiding_state_is_atomic_under_concurrency() {
        let table = Digest([7; 32]);
        let shared = Arc::new(Mutex::new(HidingState::new(table, 1, 3)));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let shared = Arc::clone(&shared);
            handles.push(std::thread::spawn(move || {
                shared.lock().unwrap().begin_charge(table, 1, 2).is_ok()
            }));
        }
        let successes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|ok| *ok)
            .count();
        assert_eq!(successes, 1);
        assert_eq!(shared.lock().unwrap().used(), 2);
    }

    #[test]
    fn recovery_keeps_the_budget_spent() {
        let mut ledger = ReadLedger::new(3);
        let _reservation = ledger.reserve(2).unwrap();
        assert_eq!(ledger.reserved(), 2);
        ledger.recover();
        assert_eq!(ledger.used(), 2);
        assert_eq!(ledger.reserved(), 0);
        assert_eq!(ledger.outstanding(), 0);
        let reservation = ledger.reserve(1).unwrap();
        ledger.cancel(reservation).unwrap();
        assert_eq!(ledger.used(), 2);
    }

    #[test]
    fn read_errors_do_not_panic() {
        let table = committed();
        let positions = [1u64, 4];
        let openings = table.open_columns(&positions).unwrap();
        let mut reads = VerifierReads::new(ledger(4));
        assert!(reads.read(&table.reader, &positions, &openings[..1]).is_err());
        let mut wrong = openings.clone();
        wrong[0].position = 2;
        assert!(reads.read(&table.reader, &positions, &wrong).is_err());
        let mut tampered = openings.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(reads.read(&table.reader, &positions, &tampered).is_err());
    }
}
