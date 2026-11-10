//! Isolated candidate for the LIN matrix shape, claims and small arithmetic.
//!
//! This module models only a candidate linear matrix claim and the three local
//! relation values used by the isolated session skeleton. It is not a complete
//! LIN proof, a model decoder, or a production relation. Challenge order, tree
//! bookkeeping, mask messages, BAT, the folding equations, the proximity tests,
//! hiding, and real seals are not connected here and remain fail-closed. The
//! trusted source of shape, claims and table identity remains a caller/plan
//! gate; a proof must not supply them. The small arithmetic tests are
//! hand-checked examples, not a soundness claim.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::field::Fp3;
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::{hash_labeled, tag_word, STATEMENT};

/// M has `matrix_rows` effective rows and `2^k_col` columns. A table has
/// `table_rows = 2^v` rows and `table_columns` entries per row; entry
/// `(y, j + 2^k_lo r)` of M is message entry `j + 2^k_lo y` of table row `r`.
/// Rows at or above `matrix_rows` are padding and are not read by the relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinShape {
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub k_col: u32,
    pub table_rows: u64,
    pub table_columns: u64,
}

impl LinShape {
    pub fn new(
        matrix_rows: u64,
        k_lo: u32,
        table_rows: u64,
        table_columns: u64,
    ) -> Result<Self> {
        if matrix_rows == 0
            || matrix_rows > (1u64 << 30)
            || k_lo > 30
            || table_rows == 0
            || !table_rows.is_power_of_two()
            || table_columns == 0
        {
            return Err(Error::InvalidConfig);
        }
        let table_log = table_rows.trailing_zeros();
        let k_col = k_lo.checked_add(table_log).ok_or(Error::InvalidConfig)?;
        let shape = Self {
            matrix_rows,
            k_lo,
            k_col,
            table_rows,
            table_columns,
        };
        shape.check()?;
        Ok(shape)
    }

    /// Revalidate every invariant at a public entry point. The fields stay
    /// inspectable for tests, but a forged or mutated shape must fail closed.
    pub fn check(&self) -> Result<()> {
        if self.matrix_rows == 0
            || self.matrix_rows > (1u64 << 30)
            || self.k_lo > 30
            || self.table_rows == 0
            || !self.table_rows.is_power_of_two()
            || self.table_columns == 0
        {
            return Err(Error::InvalidConfig);
        }
        let table_log = self.table_rows.trailing_zeros();
        let expected_k_col = self
            .k_lo
            .checked_add(table_log)
            .ok_or(Error::InvalidConfig)?;
        if expected_k_col > 60 || self.k_col != expected_k_col {
            return Err(Error::InvalidConfig);
        }
        let required = self
            .matrix_rows
            .checked_shl(self.k_lo)
            .ok_or(Error::ResourceLimit)?;
        if required > self.table_columns {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    pub fn k_row(&self) -> u32 {
        if self.matrix_rows <= 1 {
            0
        } else {
            u64::BITS - (self.matrix_rows - 1).leading_zeros()
        }
    }

    pub fn matrix_columns(&self) -> Result<u64> {
        self.check()?;
        1u64.checked_shl(self.k_col).ok_or(Error::ResourceLimit)
    }

    pub fn table_columns_used(&self) -> Result<u64> {
        self.check()?;
        self.matrix_rows
            .checked_shl(self.k_lo)
            .ok_or(Error::ResourceLimit)
    }
}

/// A table message set that M reads. All entries are public candidate data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinTable {
    pub shape: LinShape,
    pub rows: Vec<Vec<Fp3>>,
}

impl LinTable {
    pub fn new(shape: LinShape, rows: Vec<Vec<Fp3>>) -> Result<Self> {
        shape.check()?;
        let expected_rows = usize::try_from(shape.table_rows).map_err(|_| Error::ResourceLimit)?;
        let expected_cols =
            usize::try_from(shape.table_columns).map_err(|_| Error::ResourceLimit)?;
        if rows.len() != expected_rows || rows.iter().any(|row| row.len() != expected_cols) {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { shape, rows })
    }

    /// M[y, col] for an effective matrix row and a matrix column.
    pub fn at(&self, row: u64, column: u64) -> Result<Fp3> {
        self.shape.check()?;
        if row >= self.shape.matrix_rows || column >= self.shape.matrix_columns()? {
            return Err(Error::InvalidIndex);
        }
        let mask = (1u64 << self.shape.k_lo) - 1;
        let table_row = column >> self.shape.k_lo;
        let place = column & mask;
        let entry = place
            .checked_add(row.checked_shl(self.shape.k_lo).ok_or(Error::ResourceLimit)?)
            .ok_or(Error::ResourceLimit)?;
        if entry >= self.shape.table_columns {
            return Err(Error::InvalidIndex);
        }
        self.rows
            .get(table_row as usize)
            .and_then(|values| values.get(entry as usize))
            .copied()
            .ok_or(Error::InvalidIndex)
    }

    /// W = gᵀ M. `weights` has one entry for each effective M row.
    pub fn w(&self, weights: &[Fp3]) -> Result<Vec<Fp3>> {
        self.shape.check()?;
        if usize::try_from(self.shape.matrix_rows).ok() != Some(weights.len()) {
            return Err(Error::ShapeMismatch);
        }
        let columns =
            usize::try_from(self.shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
        let mut out = Vec::with_capacity(columns);
        for column in 0..columns {
            let mut sum = Fp3::ZERO;
            for (row, weight) in weights.iter().enumerate() {
                sum = sum.add(&weight.mul(&self.at(row as u64, column as u64)?));
            }
            out.push(sum);
        }
        Ok(out)
    }

    /// Omega = M eq(psi, .). `psi` has one coordinate for each matrix column.
    /// The result is dense over the row bits `k_row`; entries at or above
    /// `matrix_rows` are zero padding and are not part of the relation.
    pub fn omega(&self, psi: &[Fp3]) -> Result<Vec<Fp3>> {
        self.shape.check()?;
        if psi.len() != self.shape.k_col as usize {
            return Err(Error::ShapeMismatch);
        }
        let rows = 1usize
            .checked_shl(self.shape.k_row())
            .ok_or(Error::ResourceLimit)?;
        let columns =
            usize::try_from(self.shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
        let mut out = vec![Fp3::ZERO; rows];
        for (row, value) in out.iter_mut().enumerate().take(self.shape.matrix_rows as usize) {
            let mut sum = Fp3::ZERO;
            for column in 0..columns {
                let eq = eq_eval(psi, column as u64, self.shape.k_col)?;
                sum = sum.add(&self.at(row as u64, column as u64)?.mul(&eq));
            }
            *value = sum;
        }
        Ok(out)
    }
}

/// One owned mask entry in a claim: the entry value and its public
/// coefficient. The value must come from the request's own mask message in a
/// full session, not from a caller-chosen free field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinMask {
    pub value: Fp3,
    pub coefficient: Fp3,
}

/// A candidate public claim. `weights` are public row weights, not a secret
/// model input; `masks` are owned mask-entry values with their coefficients;
/// all fields are fixed by the request first-chain prefix before its seal in a
/// full session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinClaim {
    pub weights: Vec<Fp3>,
    pub psi: Vec<Fp3>,
    pub masks: Vec<LinMask>,
    pub target: Fp3,
}

impl LinClaim {
    /// Fixture-level shape agreement: `psi` covers `k_col` and `weights`
    /// covers the effective `matrix_rows`. This is not a real chain/claim
    /// binding and does not prove the relation or the source of either field.
    pub fn check(&self, shape: &LinShape) -> Result<()> {
        shape.check()?;
        if self.psi.len() != shape.k_col as usize
            || self.weights.len() != shape.matrix_rows as usize
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.weights.len() as u64);
        for value in &self.weights {
            writer.put_fp3(value);
        }
        writer.put_u64(self.psi.len() as u64);
        for value in &self.psi {
            writer.put_fp3(value);
        }
        writer.put_u64(self.masks.len() as u64);
        for mask in &self.masks {
            writer.put_fp3(&mask.value);
            writer.put_fp3(&mask.coefficient);
        }
        writer.put_fp3(&self.target);
        Ok(hash_labeled(
            tag_word(STATEMENT, 0),
            self.weights.len() as u64,
            None,
            &writer.into_bytes(),
        )?)
    }
}

/// The (Eval) check: W(psi) + sum(coeff * mask value) = target.
pub fn lin_eval_holds(table: &LinTable, claim: &LinClaim) -> Result<bool> {
    claim.check(&table.shape)?;
    let w = table.w(&claim.weights)?;
    let mut value = eval(&w, &claim.psi)?;
    for mask in &claim.masks {
        value = value.add(&mask.value.mul(&mask.coefficient));
    }
    Ok(value == claim.target)
}

/// The (Eval) check with a fixed submitted W. This variant checks the value at
/// the claim point directly against the submitted vector and does not recompute
/// W from the table. The caller must fix and authenticate W separately; the
/// vector's source is outside this function.
pub fn lin_eval_holds_with(
    table: &LinTable,
    claim: &LinClaim,
    w: &[Fp3],
) -> Result<bool> {
    claim.check(&table.shape)?;
    if w.len() as u64 != table.shape.matrix_columns()? {
        return Err(Error::ShapeMismatch);
    }
    let mut value = eval(w, &claim.psi)?;
    for mask in &claim.masks {
        value = value.add(&mask.value.mul(&mask.coefficient));
    }
    Ok(value == claim.target)
}

/// The (Dot) check: <weights, Omega> = W(psi*). `shared_point` is the shared
/// challenge point, distinct from the request's `claim.psi` used by Eval.
/// This convenience recomputes Omega at that point; it cannot by itself reject
/// a substituted Omega. A verifier that gets Omega from a committed message
/// must use the explicit form below.
pub fn lin_dot_holds(
    table: &LinTable,
    claim: &LinClaim,
    shared_point: &[Fp3],
) -> Result<bool> {
    let omega = table.omega(shared_point)?;
    lin_dot_holds_with(table, claim, shared_point, &omega)
}

/// The (Dot) check with an explicitly supplied shared Omega and shared point.
/// The caller's source of Omega is a committed message, not this function; the
/// explicit form is what lets a tampered Omega or a wrong point be rejected.
pub fn lin_dot_holds_with(
    table: &LinTable,
    claim: &LinClaim,
    shared_point: &[Fp3],
    omega: &[Fp3],
) -> Result<bool> {
    table.shape.check()?;
    claim.check(&table.shape)?;
    let expected = power_of_two_len(table.shape.k_row() as usize)?;
    if claim.weights.len() != table.shape.matrix_rows as usize
        || shared_point.len() != table.shape.k_col as usize
        || omega.len() != expected
    {
        return Err(Error::ShapeMismatch);
    }
    let w = table.w(&claim.weights)?;
    let mut left = Fp3::ZERO;
    for (weight, value) in claim
        .weights
        .iter()
        .zip(omega.iter().take(table.shape.matrix_rows as usize))
    {
        left = left.add(&weight.mul(value));
    }
    Ok(left == eval(&w, shared_point)?)
}

/// The (Dot) check with a fixed submitted W and shared Omega. The right side
/// uses the submitted W directly; this variant does not recompute W from the
/// table. The caller must fix and authenticate W separately.
pub fn lin_dot_holds_with_w(
    table: &LinTable,
    claim: &LinClaim,
    shared_point: &[Fp3],
    omega: &[Fp3],
    w: &[Fp3],
) -> Result<bool> {
    table.shape.check()?;
    claim.check(&table.shape)?;
    let expected = power_of_two_len(table.shape.k_row() as usize)?;
    if claim.weights.len() != table.shape.matrix_rows as usize
        || shared_point.len() != table.shape.k_col as usize
        || omega.len() != expected
        || w.len() as u64 != table.shape.matrix_columns()?
    {
        return Err(Error::ShapeMismatch);
    }
    let mut left = Fp3::ZERO;
    for (weight, value) in claim
        .weights
        .iter()
        .zip(omega.iter().take(table.shape.matrix_rows as usize))
    {
        left = left.add(&weight.mul(value));
    }
    Ok(left == eval(w, shared_point)?)
}

/// The (Fold) check: Omega at zeta equals the folded-row inner product.
/// `omega` is dense over `k_row` bits; `folded` is dense over the `k_lo` column
/// place and `k_row` row bits, with index `j + 2^k_lo y`; `psi_lo` is the first
/// `k_lo` shared-point coordinates.
pub fn lin_fold_holds(
    shape: LinShape,
    omega: &[Fp3],
    psi_lo: &[Fp3],
    zeta: &[Fp3],
    folded: &[Fp3],
) -> Result<bool> {
    shape.check()?;
    let folded_bits = shape
        .k_lo
        .checked_add(shape.k_row())
        .ok_or(Error::ResourceLimit)? as usize;
    if psi_lo.len() != shape.k_lo as usize
        || zeta.len() != shape.k_row() as usize
        || omega.len() != power_of_two_len(shape.k_row() as usize)?
        || folded.len() != power_of_two_len(folded_bits)?
    {
        return Err(Error::ShapeMismatch);
    }
    let left = eval_truncated(omega, zeta, shape.matrix_rows)?;
    let mut right = Fp3::ZERO;
    for row in 0..shape.matrix_rows {
        let row_weight = eq_eval(zeta, row, shape.k_row())?;
        for place in 0..(1u64 << shape.k_lo) {
            let entry = place + (row << shape.k_lo);
            let weight = row_weight.mul(&eq_eval(psi_lo, place, shape.k_lo)?);
            right = right.add(&weight.mul(&folded[entry as usize]));
        }
    }
    Ok(left == right)
}

/// Multilinear evaluation of a power-of-two length vector.
pub fn eval(values: &[Fp3], point: &[Fp3]) -> Result<Fp3> {
    if values.len() != power_of_two_len(point.len())? {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate() {
        sum = sum.add(&value.mul(&eq_eval(point, index as u64, point.len() as u32)?));
    }
    Ok(sum)
}

fn power_of_two_len(bits: usize) -> Result<usize> {
    let shift = u32::try_from(bits).map_err(|_| Error::ResourceLimit)?;
    1usize.checked_shl(shift).ok_or(Error::ResourceLimit)
}

/// Multilinear evaluation with rows at or above `effective_rows` treated as
/// padding (the small analogue of a truncated matrix row space).
pub fn eval_truncated(values: &[Fp3], point: &[Fp3], effective_rows: u64) -> Result<Fp3> {
    if values.len() != power_of_two_len(point.len())?
        || effective_rows as usize > values.len()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut sum = Fp3::ZERO;
    for (index, value) in values.iter().enumerate().take(effective_rows as usize) {
        sum = sum.add(&value.mul(&eq_eval(point, index as u64, point.len() as u32)?));
    }
    Ok(sum)
}

fn eq_eval(point: &[Fp3], index: u64, bits: u32) -> Result<Fp3> {
    if point.len() != bits as usize {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ONE;
    for (bit, coordinate) in point.iter().enumerate() {
        let one_minus = Fp3::ONE.sub(coordinate);
        value = value.mul(if (index >> bit) & 1 == 1 {
            coordinate
        } else {
            &one_minus
        });
    }
    Ok(value)
}

/// The public parameters a fixture chain carries: the matrix/table shape, the
/// code setup, and the table identity, record and configuration. The fixture
/// compares these values across chains; it does not prove that a real statement
/// or table supplies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinPublicParameters {
    pub shape: LinShape,
    pub codes: Digest,
    pub table_identity: Digest,
    pub table_record: Digest,
    pub configuration: Digest,
}

impl LinPublicParameters {
    pub fn check(&self) -> Result<()> {
        self.shape.check()?;
        if self.codes == Digest::ZERO
            || self.table_identity == Digest::ZERO
            || self.table_record == Digest::ZERO
            || self.configuration == Digest::ZERO
        {
            return Err(Error::InvalidConfig);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest> {
        self.check()?;
        let mut writer = CanonicalWriter::new();
        put_public(&mut writer, self);
        Ok(hash_labeled(
            tag_word(STATEMENT, 0),
            0,
            None,
            &writer.into_bytes(),
        )?)
    }
}

/// Fixture-only first-chain seal. The digest covers the fixed claim and the
/// structured public parameters; it is not a real seal round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinFirstFixture {
    pub claim: LinClaim,
    pub public: LinPublicParameters,
    pub first_query: Vec<u8>,
    pub seal: Digest,
}

impl LinFirstFixture {
    pub fn new(claim: LinClaim, public: LinPublicParameters) -> Result<Self> {
        public.check()?;
        claim.check(&public.shape)?;
        let claim_digest = claim.digest()?;
        let first_query = first_prefix(&claim_digest, &public)?;
        let seal = fixture_seal(&claim_digest, &first_query);
        if seal == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        Ok(Self {
            claim,
            public,
            first_query,
            seal,
        })
    }
}

/// Fixture-only shared-chain snapshot carrying the binding root and the same
/// structured public parameters. No real seal, tree, or challenge round is
/// claimed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinSharedFixture {
    pub binding_root: Digest,
    pub public: LinPublicParameters,
    pub first_query: Vec<u8>,
    pub seal: Digest,
}

impl LinSharedFixture {
    pub fn new(first_seals: &[Digest], public: LinPublicParameters) -> Result<Self> {
        public.check()?;
        if first_seals.is_empty() || first_seals.iter().any(|seal| *seal == Digest::ZERO) {
            return Err(Error::InvalidSessionState);
        }
        let binding_root = binding_root(first_seals);
        if binding_root == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        let first_query = shared_prefix(&binding_root, &public)?;
        let seal = fixture_seal(&binding_root, &first_query);
        if seal == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        Ok(Self {
            binding_root,
            public,
            first_query,
            seal,
        })
    }
}

/// Fixture-only second-chain snapshot. It follows one request's first seal and
/// the shared seal; the bytes are a candidate binding, not a real chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinSecondFixture {
    pub own_seal: Digest,
    pub shared_seal: Digest,
    pub public: LinPublicParameters,
    pub follows: Vec<Digest>,
    pub first_query: Vec<u8>,
    pub seal: Digest,
}

impl LinSecondFixture {
    pub fn new(own: &LinFirstFixture, shared: &LinSharedFixture) -> Result<Self> {
        if own.seal == Digest::ZERO || shared.seal == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        let mut writer = CanonicalWriter::new();
        writer.put_digest(&own.seal);
        writer.put_digest(&shared.seal);
        let first_query = writer.into_bytes();
        let seal = fixture_seal(&shared.binding_root, &first_query);
        if seal == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        Ok(Self {
            own_seal: own.seal,
            shared_seal: shared.seal,
            public: shared.public,
            follows: vec![own.seal, shared.seal],
            first_query,
            seal,
        })
    }
}

/// Check a fixture skeleton: every first seal is the fixture seal of its fixed
/// claim and structured public prefix; all first and shared and second public
/// parameters agree; the binding root is the digest of all first seals; the
/// shared prefix carries the binding root and public parameters; the second
/// prefix follows its own first and the shared seal. This is a structural
/// fixture check, not a chain protocol or a real public-parameter proof.
pub fn check_lin_skeleton(
    first: &[LinFirstFixture],
    shared: &LinSharedFixture,
    second: &LinSecondFixture,
) -> Result<()> {
    if first.is_empty() {
        return Err(Error::StatementMismatch);
    }
    shared.public.check()?;
    second.public.check()?;
    for entry in first {
        entry.public.check()?;
        entry.claim.check(&entry.public.shape)?;
        let claim_digest = entry.claim.digest()?;
        if entry.seal == Digest::ZERO
            || entry.first_query.is_empty()
            || entry.public != shared.public
            || entry.first_query != first_prefix(&claim_digest, &entry.public)?
            || entry.seal != fixture_seal(&claim_digest, &entry.first_query)
        {
            return Err(Error::StatementMismatch);
        }
    }
    let expected_root = binding_root(&first.iter().map(|entry| entry.seal).collect::<Vec<_>>());
    if shared.binding_root != expected_root
        || shared.first_query != shared_prefix(&shared.binding_root, &shared.public)?
        || shared.seal != fixture_seal(&shared.binding_root, &shared.first_query)
    {
        return Err(Error::StatementMismatch);
    }
    if second.public != shared.public
        || second.own_seal == Digest::ZERO
        || second.shared_seal == Digest::ZERO
        || second.follows != vec![second.own_seal, second.shared_seal]
        || !first.iter().any(|entry| entry.seal == second.own_seal)
        || second.shared_seal != shared.seal
    {
        return Err(Error::StatementMismatch);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_digest(&second.own_seal);
    writer.put_digest(&second.shared_seal);
    let expected_query = writer.into_bytes();
    if second.first_query != expected_query
        || second.seal != fixture_seal(&shared.binding_root, &second.first_query)
    {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn put_public(writer: &mut CanonicalWriter, public: &LinPublicParameters) {
    writer.put_u64(public.shape.matrix_rows);
    writer.put_u64(public.shape.k_lo as u64);
    writer.put_u64(public.shape.k_col as u64);
    writer.put_u64(public.shape.table_rows);
    writer.put_u64(public.shape.table_columns);
    writer.put_digest(&public.codes);
    writer.put_digest(&public.table_identity);
    writer.put_digest(&public.table_record);
    writer.put_digest(&public.configuration);
}

fn first_prefix(claim: &Digest, public: &LinPublicParameters) -> Result<Vec<u8>> {
    public.check()?;
    let mut writer = CanonicalWriter::new();
    writer.put_digest(claim);
    put_public(&mut writer, public);
    Ok(writer.into_bytes())
}

fn shared_prefix(binding_root: &Digest, public: &LinPublicParameters) -> Result<Vec<u8>> {
    public.check()?;
    let mut writer = CanonicalWriter::new();
    writer.put_digest(binding_root);
    put_public(&mut writer, public);
    Ok(writer.into_bytes())
}

/// Fixture-only linear hash over first seals. It is not a real binding tree
/// root and does not prove a sealed-chain or public-parameter relation.
fn binding_root(seals: &[Digest]) -> Digest {
    let mut writer = CanonicalWriter::new();
    for seal in seals {
        writer.put_digest(seal);
    }
    fixture_digest(&writer.into_bytes())
}

fn fixture_seal(start: &Digest, bytes: &[u8]) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_digest(start);
    writer.put_bytes(bytes);
    fixture_digest(&writer.into_bytes())
}

fn fixture_digest(bytes: &[u8]) -> Digest {
    hash_labeled(tag_word(STATEMENT, 0), 0, None, bytes).unwrap_or(Digest::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> LinShape {
        LinShape::new(3, 2, 2, 12).unwrap()
    }

    fn table() -> LinTable {
        let rows = (0..2)
            .map(|row| {
                (0..12)
                    .map(|column| Fp3::from_u64((row * 100 + column) as u64 + 1))
                    .collect()
            })
            .collect();
        LinTable::new(shape(), rows).unwrap()
    }

    fn public() -> LinPublicParameters {
        LinPublicParameters {
            shape: shape(),
            codes: Digest([1; 32]),
            table_identity: Digest([2; 32]),
            table_record: Digest([3; 32]),
            configuration: Digest([4; 32]),
        }
    }

    #[test]
    fn shape_mapping_and_non_power_rows_are_exact() {
        let shape = shape();
        assert_eq!(shape.k_row(), 2);
        assert_eq!(shape.k_col, 3);
        assert_eq!(shape.matrix_columns().unwrap(), 8);
        assert_eq!(shape.table_columns_used().unwrap(), 12);
        let table = table();
        assert_eq!(table.at(0, 0).unwrap(), Fp3::from_u64(1));
        assert_eq!(table.at(0, 5).unwrap(), Fp3::from_u64(102));
        assert_eq!(table.at(1, 0).unwrap(), Fp3::from_u64(5));
        assert_eq!(table.at(2, 0).unwrap(), Fp3::from_u64(9));
        assert_eq!(table.at(0, 7).unwrap(), Fp3::from_u64(104));
        assert_eq!(table.at(1, 7).unwrap(), Fp3::from_u64(108));
        assert_eq!(table.at(2, 7).unwrap(), Fp3::from_u64(112));
        assert!(table.at(3, 0).is_err());
        assert!(table.at(0, 8).is_err());
        assert!(LinShape::new(0, 2, 2, 12).is_err());
        assert!(LinShape::new(3, 2, 3, 12).is_err());
        assert!(LinShape::new(3, 2, 2, 11).is_err());
        assert!(LinShape::new(1, 31, 1, 1).is_err());
        assert!(LinShape::new(1, 0, 1u64 << 61, 1).is_err());
        assert!(LinShape::new(1 << 30, 0, 1, 1).is_err());
        assert!(LinShape::new((1u64 << 30) + 1, 0, 1, 1).is_err());
    }

    #[test]
    fn forged_shapes_fail_closed_at_public_entries() {
        let mut forged = table();
        forged.shape.k_lo = 31;
        assert!(forged.at(0, 0).is_err());
        assert!(forged.w(&[Fp3::ZERO; 3]).is_err());
        assert!(forged.omega(&[Fp3::ZERO; 3]).is_err());
        assert!(forged.shape.matrix_columns().is_err());
        assert!(forged.shape.table_columns_used().is_err());
        forged.shape.k_lo = 64;
        assert!(forged.at(0, 0).is_err());
        let mut bad_kcol = table();
        bad_kcol.shape.k_col = 64;
        assert!(bad_kcol.at(0, 0).is_err());
        assert!(bad_kcol.omega(&[Fp3::ZERO; 3]).is_err());

        let mut mismatch = table();
        mismatch.shape.k_col = 4;
        assert!(mismatch.at(0, 0).is_err());
        assert!(mismatch.shape.check().is_err());

        let mut non_power = table();
        non_power.shape.table_rows = 3;
        assert!(non_power.at(0, 0).is_err());

        let mut huge = shape();
        huge.matrix_rows = u64::MAX;
        assert!(huge.check().is_err());
        assert!(huge.k_row() > 0);
        assert!(huge.matrix_columns().is_err());

        let rows = table().rows;
        assert!(LinTable::new(
            LinShape {
                matrix_rows: 3,
                k_lo: 2,
                k_col: 4,
                table_rows: 2,
                table_columns: 12,
            },
            rows
        )
        .is_err());

        let mut short_row = table();
        short_row.rows[1].pop();
        assert!(short_row.at(2, 7).is_err());
    }

    #[test]
    fn evaluation_rejects_dimension_overflow_and_accepts_zero_variables() {
        let scalar = vec![Fp3::from_u64(7)];
        assert_eq!(eval(&scalar, &[]).unwrap(), scalar[0]);
        assert_eq!(eval_truncated(&scalar, &[], 1).unwrap(), scalar[0]);
        assert!(eval(&[], &[]).is_err());
        let long = vec![Fp3::ZERO; usize::BITS as usize + 1];
        assert!(eval(&[], &long).is_err());
        assert!(eval_truncated(&[], &long, 0).is_err());
        let machine_width = vec![Fp3::ZERO; usize::BITS as usize];
        assert!(eval(&[], &machine_width).is_err());
        assert!(eval_truncated(&[], &machine_width, 0).is_err());
    }

    #[test]
    fn w_omega_and_dot_are_independently_checked() {
        let table = table();
        let weights = vec![Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(5)];
        let psi = vec![Fp3::from_u64(7), Fp3::from_u64(11), Fp3::from_u64(13)];
        let w = table.w(&weights).unwrap();
        assert_eq!(
            w,
            vec![
                Fp3::from_u64(62),
                Fp3::from_u64(72),
                Fp3::from_u64(82),
                Fp3::from_u64(92),
                Fp3::from_u64(1062),
                Fp3::from_u64(1072),
                Fp3::from_u64(1082),
                Fp3::from_u64(1092),
            ]
        );
        let request_omega = table.omega(&psi).unwrap();
        assert_eq!(
            request_omega,
            vec![
                Fp3::from_u64(1330),
                Fp3::from_u64(1334),
                Fp3::from_u64(1338),
                Fp3::ZERO,
            ]
        );
        for row in 0..3usize {
            let mut expected = Fp3::ZERO;
            for column in 0..8u64 {
                let eq = eq_eval(&psi, column, 3).unwrap();
                expected = expected.add(&table.at(row as u64, column).unwrap().mul(&eq));
            }
            assert_eq!(request_omega[row], expected);
        }
        let shared_point = vec![Fp3::from_u64(17), Fp3::from_u64(19), Fp3::from_u64(23)];
        assert_ne!(shared_point, psi);
        let shared_omega = table.omega(&shared_point).unwrap();
        assert_eq!(
            shared_omega,
            vec![
                Fp3::from_u64(2356),
                Fp3::from_u64(2360),
                Fp3::from_u64(2364),
                Fp3::ZERO,
            ]
        );
        assert_eq!(eval(&w, &psi).unwrap(), Fp3::from_u64(13352));
        assert_eq!(eval(&w, &shared_point).unwrap(), Fp3::from_u64(23612));
        assert_eq!(
            weights[0]
                .mul(&shared_omega[0])
                .add(&weights[1].mul(&shared_omega[1]))
                .add(&weights[2].mul(&shared_omega[2])),
            Fp3::from_u64(23612)
        );
        let mask = LinMask {
            value: Fp3::from_u64(17),
            coefficient: Fp3::from_u64(3),
        };
        let target = eval(&w, &psi).unwrap().add(&mask.value.mul(&mask.coefficient));
        let mut claim = LinClaim {
            weights,
            psi: psi.clone(),
            masks: vec![mask],
            target,
        };
        assert!(lin_eval_holds(&table, &claim).unwrap());
        assert!(lin_dot_holds(&table, &claim, &shared_point).unwrap());
        assert!(lin_dot_holds_with(&table, &claim, &shared_point, &shared_omega).unwrap());
        assert!(!lin_dot_holds_with(&table, &claim, &claim.psi, &shared_omega).unwrap());
        let mut bad_omega = shared_omega.clone();
        bad_omega[0] = bad_omega[0].add(&Fp3::from_u64(1));
        assert!(!lin_dot_holds_with(&table, &claim, &shared_point, &bad_omega).unwrap());
        assert!(lin_dot_holds_with(&table, &claim, &shared_point, &shared_omega[..3]).is_err());
        claim.target = target.add(&Fp3::from_u64(1));
        assert!(!lin_eval_holds(&table, &claim).unwrap());
        claim.target = target;
        claim.masks[0].coefficient = Fp3::from_u64(6);
        assert!(!lin_eval_holds(&table, &claim).unwrap());
        claim.masks[0].coefficient = Fp3::from_u64(3);
        claim.psi[0] = Fp3::from_u64(19);
        assert!(!lin_eval_holds(&table, &claim).unwrap());
        assert!(lin_dot_holds_with(&table, &claim, &shared_point, &shared_omega).unwrap());
        claim.psi[0] = Fp3::from_u64(7);
        let mut wrong_weights = claim.clone();
        wrong_weights.weights[0] = Fp3::from_u64(9);
        assert!(!lin_eval_holds(&table, &wrong_weights).unwrap());
        let mut short_weights = claim.clone();
        short_weights.weights.pop();
        assert!(lin_eval_holds(&table, &short_weights).is_err());
        assert!(lin_dot_holds_with(&table, &short_weights, &shared_point, &shared_omega).is_err());
        let mut short_psi = claim.clone();
        short_psi.psi.pop();
        assert!(short_psi.check(&table.shape).is_err());
        assert!(lin_eval_holds(&table, &short_psi).is_err());
        assert!(eval(&w, &psi[..2]).is_err());
    }

    #[test]
    fn fold_check_is_hand_checked() {
        let table = table();
        let psi = vec![Fp3::from_u64(2), Fp3::from_u64(3), Fp3::from_u64(5)];
        let zeta = vec![Fp3::from_u64(7), Fp3::from_u64(11)];
        let omega = table.omega(&psi).unwrap();
        let psi_high = psi[2];
        let folded = (0..16)
            .map(|entry| {
                if entry >= 12 {
                    Fp3::from_u64(77)
                } else {
                    let zero_row = table.rows[0][entry];
                    let one_row = table.rows[1][entry];
                    zero_row
                        .mul(&Fp3::ONE.sub(&psi_high))
                        .add(&one_row.mul(&psi_high))
                }
            })
            .collect::<Vec<_>>();
        let mut right = Fp3::ZERO;
        for row in 0..3u64 {
            let row_weight = eq_eval(&zeta, row, 2).unwrap();
            for place in 0..4u64 {
                let entry = place + (row << 2);
                right = right.add(
                    &row_weight
                        .mul(&eq_eval(&psi[..2], place, 2).unwrap())
                        .mul(&folded[entry as usize]),
                );
            }
        }
        let left = eval_truncated(&omega, &zeta, 3).unwrap();
        assert_eq!(left, right);
        assert!(lin_fold_holds(shape(), &omega, &psi[..2], &zeta, &folded).unwrap());
        let mut padded_omega = omega.clone();
        padded_omega[3] = Fp3::from_u64(42);
        assert!(lin_fold_holds(shape(), &padded_omega, &psi[..2], &zeta, &folded).unwrap());
        assert!(lin_fold_holds(shape(), &omega, &psi[..1], &zeta, &folded).is_err());
        let mut changed_omega = omega.clone();
        changed_omega[0] = changed_omega[0].add(&Fp3::from_u64(1));
        assert!(!lin_fold_holds(shape(), &changed_omega, &psi[..2], &zeta, &folded).unwrap());
        let mut short = folded.clone();
        short.pop();
        assert!(lin_fold_holds(shape(), &omega, &psi[..2], &zeta, &short).is_err());
        let mut different = folded;
        different[0] = Fp3::from_u64(99);
        assert!(!lin_fold_holds(shape(), &omega, &psi[..2], &zeta, &different).unwrap());
    }

    #[test]
    fn claim_encoding_is_stable_and_public() {
        let claim = LinClaim {
            weights: vec![Fp3::from_u64(1)],
            psi: vec![Fp3::from_u64(2)],
            masks: Vec::new(),
            target: Fp3::from_u64(3),
        };
        assert_eq!(claim.digest().unwrap(), claim.clone().digest().unwrap());
        assert_ne!(
            claim.digest().unwrap(),
            LinClaim {
                target: Fp3::from_u64(4),
                ..claim
            }
            .digest()
            .unwrap()
        );
    }

    #[test]
    fn skeleton_requires_every_first_seal_and_matching_public_parameters() {
        let claim = |target: u64| LinClaim {
            weights: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)],
            psi: vec![Fp3::from_u64(4), Fp3::from_u64(5), Fp3::from_u64(6)],
            masks: Vec::new(),
            target: Fp3::from_u64(target),
        };
        let first_a = LinFirstFixture::new(claim(3), public()).unwrap();
        let first_b = LinFirstFixture::new(claim(4), public()).unwrap();
        let shared = LinSharedFixture::new(&[first_a.seal, first_b.seal], public()).unwrap();
        let second = LinSecondFixture::new(&first_b, &shared).unwrap();
        check_lin_skeleton(&[first_a.clone(), first_b.clone()], &shared, &second).unwrap();
        assert!(check_lin_skeleton(&[first_b.clone()], &shared, &second).is_err());

        let mut wrong_identity = shared.clone();
        wrong_identity.public.table_identity = Digest([8; 32]);
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &wrong_identity,
            &second
        )
        .is_err());
        let mut wrong_codes = shared.clone();
        wrong_codes.public.codes = Digest([8; 32]);
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &wrong_codes,
            &second
        )
        .is_err());
        let mut wrong_record = shared.clone();
        wrong_record.public.table_record = Digest([8; 32]);
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &wrong_record,
            &second
        )
        .is_err());
        let mut wrong_configuration = shared.clone();
        wrong_configuration.public.configuration = Digest([8; 32]);
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &wrong_configuration,
            &second
        )
        .is_err());

        let mut wrong_first_public = first_a.clone();
        wrong_first_public.public.configuration = Digest([8; 32]);
        assert!(check_lin_skeleton(
            &[wrong_first_public, first_b.clone()],
            &shared,
            &second
        )
        .is_err());
        let mut wrong_claim_shape = first_a.clone();
        wrong_claim_shape.claim.psi.pop();
        assert!(check_lin_skeleton(
            &[wrong_claim_shape, first_b.clone()],
            &shared,
            &second
        )
        .is_err());
        let mut bad_claim = LinClaim {
            weights: vec![Fp3::from_u64(1)],
            psi: vec![Fp3::from_u64(2)],
            masks: Vec::new(),
            target: Fp3::from_u64(3),
        };
        assert!(LinFirstFixture::new(bad_claim.clone(), public()).is_err());
        bad_claim.psi = vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)];
        assert!(LinFirstFixture::new(bad_claim, public()).is_err());
        assert!(LinFirstFixture::new(
            LinClaim {
                weights: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)],
                psi: vec![Fp3::from_u64(1), Fp3::from_u64(2)],
                masks: Vec::new(),
                target: Fp3::from_u64(3),
            },
            public()
        )
        .is_err());
        let mut wrong_second_public = second.clone();
        wrong_second_public.public.shape = LinShape::new(5, 2, 2, 20).unwrap();
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &shared,
            &wrong_second_public
        )
        .is_err());

        let mut empty_query = first_a.clone();
        empty_query.first_query.clear();
        assert!(check_lin_skeleton(&[empty_query], &shared, &second).is_err());
        let mut tampered_shared_query = shared.clone();
        tampered_shared_query.first_query.push(0);
        assert!(check_lin_skeleton(
            &[first_a.clone(), first_b.clone()],
            &tampered_shared_query,
            &second
        )
        .is_err());

        assert!(LinSharedFixture::new(&[], public()).is_err());
        assert!(LinSharedFixture::new(&[Digest::ZERO], public()).is_err());
        let mut zero_identity = public();
        zero_identity.table_identity = Digest::ZERO;
        assert!(LinSharedFixture::new(&[first_a.seal], zero_identity).is_err());
        assert!(LinFirstFixture::new(claim(5), zero_identity).is_err());
        let mut zero_codes = public();
        zero_codes.codes = Digest::ZERO;
        assert!(LinFirstFixture::new(claim(6), zero_codes).is_err());
    }

    #[test]
    fn skeleton_is_a_fixture_not_a_chain_claim() {
        let claim = LinClaim {
            weights: vec![Fp3::from_u64(1), Fp3::from_u64(2), Fp3::from_u64(3)],
            psi: vec![Fp3::from_u64(4), Fp3::from_u64(5), Fp3::from_u64(6)],
            masks: Vec::new(),
            target: Fp3::from_u64(3),
        };
        let first = LinFirstFixture::new(claim, public()).unwrap();
        let shared = LinSharedFixture::new(&[first.seal], public()).unwrap();
        let second = LinSecondFixture::new(&first, &shared).unwrap();
        check_lin_skeleton(&[first.clone()], &shared, &second).unwrap();
        assert!(check_lin_skeleton(&[], &shared, &second).is_err());
        let mut missing = second.clone();
        missing.follows.pop();
        assert!(check_lin_skeleton(&[first.clone()], &shared, &missing).is_err());
        let mut bad_root = shared.clone();
        bad_root.binding_root = Digest([9; 32]);
        assert!(check_lin_skeleton(&[first.clone()], &bad_root, &second).is_err());
        let mut changed_claim = first.clone();
        changed_claim.claim.target = Fp3::from_u64(99);
        assert!(check_lin_skeleton(&[changed_claim], &shared, &second).is_err());
        let mut wrong_own = second.clone();
        wrong_own.own_seal = Digest([8; 32]);
        wrong_own.follows[0] = Digest([8; 32]);
        assert!(check_lin_skeleton(&[first.clone()], &shared, &wrong_own).is_err());
        let mut label_only = first;
        label_only.seal = Digest([7; 32]);
        assert!(check_lin_skeleton(&[label_only], &shared, &second).is_err());
    }
}
