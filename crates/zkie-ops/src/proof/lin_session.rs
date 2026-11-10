//! Three-stage LIN request component.
//!
//! The first stage fixes the matrix witness `W` and the per-row weights, draws
//! the evaluation point `psi` from the first chain, and checks the evaluation
//! relation (including mask entries) and the full-value `W` control. The shared
//! stage resumes from the first chain's final value, draws a distinct shared
//! point, fixes `Omega` and the folded message, draws `zeta`, and checks the dot
//! relation with the explicit `Omega` and the full-value `Omega` control. The
//! second stage starts a chain whose first query binds the first chain's first
//! query and the shared chain's final value (the shared seal), fixes the final
//! transcript, and checks the fold relation with the same `Omega` and folded
//! messages.
//!
//! The table `M` is built from an authenticated asset opening: the trusted
//! model anchor is verified first, then the opening is checked with the
//! existing Merkle opener, including asset id, root, row order and width. The
//! Eval and Dot relations use the submitted W vector directly; they do not
//! recompute W. The row-wise controls recompute W and Omega from the table and
//! compare every submitted value.
//!
//! The weights, masks, claimed evaluation, W message, Omega message and folded
//! rows are self-reported values. W and Omega openings only show membership in
//! the commitments absorbed here; they do not show that those commitments are
//! the output of a trusted graph predecessor. The upstream graph source `x` is
//! not closed in this step. Authentication of the folded rows, commitment
//! consistency, proximity, hiding and single-shot semantics remain unconnected.
//!
//! The base field has `p < 2^64` and `Fp3` is a cubic extension, so
//! `|Fp3| = p^3 < 2^192`. This demonstration gives no soundness bound: a
//! multivariate nonzero polynomial detection bound needs its degree and the
//! challenge distribution, and Omega is chosen after the shared challenge, so
//! a bound fixed before that point does not apply. Only local equality checks
//! and the deterministic full-value controls are run.
//!
//! The binding is assembled by the caller with the existing binding candidate.
//! The candidate protocol identity and request parameters are demo
//! placeholders, not real protocol values. The tests use deterministic
//! fixtures; they do not seed a production session. The field order, messages
//! and challenge positions are candidate-internal and replaceable.

use zkie_core::common::codec::DecodeLimits;
use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::chain::{Chain, ChainMessage, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::{Fp3, FP3_BYTES};
use zkie_engine::pcs::request::lin::{
    eval, lin_dot_holds_with_w, lin_eval_holds_with, lin_fold_holds, LinClaim, LinMask, LinShape,
    LinTable,
};
use zkie_engine::pcs::request::params::{
    FINAL_CHAIN_LABEL, REQUEST_CHAIN_LABEL, SHARED_CHAIN_LABEL,
};
use zkie_engine::primitives::codec::{CanonicalReader, CanonicalWriter};

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleId, OracleOpening, OracleRegistry,
};
use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelCommitmentPublic,
};
use crate::proof::session_baseline::TrustedModel;

const MESSAGE_SCHEMA_VERSION: u64 = 1;

const STAGE_FIRST: u64 = 1;
const STAGE_SHARED: u64 = 2;
const STAGE_SECOND: u64 = 3;

const FIRST_PHASE_COMMIT: u64 = 1;
const FIRST_PHASE_CLAIM: u64 = 2;
const SHARED_PHASE_START: u64 = 1;
const SHARED_PHASE_OMEGA_FOLD: u64 = 2;
const SECOND_PHASE_FOLD: u64 = 1;

const W_LABEL: Digest = Digest([0x71; 32]);
const WEIGHTS_LABEL: Digest = Digest([0x74; 32]);
const OMEGA_LABEL: Digest = Digest([0x72; 32]);
const FOLDED_LABEL: Digest = Digest([0x73; 32]);

/// A self-reported committed vector. The openings prove membership in the
/// committed root; they do not authenticate the source of the values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinOracle {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
    pub values: Vec<Fp3>,
    pub openings: Vec<OracleOpening>,
}

/// First stage: the fixed W and weight commitments, the drawn evaluation point,
/// and the claimed evaluation with its mask entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinFirstMessage {
    pub w: LinOracle,
    pub weights: LinOracle,
    pub psi: Vec<Fp3>,
    pub masks: Vec<LinMask>,
    pub target: Fp3,
}

/// Shared stage: the first chain's final value, the drawn shared point, the
/// fixed Omega and folded commitments, and the drawn fold point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinSharedMessage {
    pub first_seal: Digest,
    pub point: Vec<Fp3>,
    pub omega: LinOracle,
    pub folded: LinOracle,
    pub zeta: Vec<Fp3>,
}

/// Second stage: the first query the second chain binds, the shared chain's
/// final value, and the second chain's own final value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinSecondMessage {
    pub first_query: Digest,
    pub shared_seal: Digest,
    pub seal: Digest,
}

/// The canonical three-stage message set. The encoding is positional and each
/// stage block carries its own stage tag, so a reordered encoding is rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinRequestMessages {
    pub first: LinFirstMessage,
    pub shared: LinSharedMessage,
    pub second: LinSecondMessage,
}

/// Verifier outcome. Every boolean is true on a successful return; the values
/// are exposed for tests and for callers that need the recomputed transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinRequestOutcome {
    pub first_query: Digest,
    pub first_seal: Digest,
    pub shared_seal: Digest,
    pub seal: Digest,
    pub psi: Vec<Fp3>,
    pub shared: Vec<Fp3>,
    pub zeta: Vec<Fp3>,
    pub eval_holds: bool,
    pub dot_holds: bool,
    pub fold_holds: bool,
    pub control_w: bool,
    pub control_omega: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinControlChecks {
    pub w_matches: bool,
    pub omega_matches: bool,
}

fn digest_words(digest: &Digest) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (index, chunk) in digest.0.chunks_exact(8).enumerate() {
        words[index] = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
    }
    words
}

/// Converts a signed fixed-point word stored as two's-complement `u64` into a
/// field element. Asset and opening values are signed integers, so an unsigned
/// reduction would change their field representation.
pub fn fp3_from_signed_word(value: u64) -> Fp3 {
    let signed = value as i64;
    let base = Fp3::from_u64(signed.unsigned_abs());
    if signed < 0 { base.neg() } else { base }
}

fn fp3_words(value: &Fp3) -> [u64; 3] {
    [
        value.c[0].as_canonical_u64(),
        value.c[1].as_canonical_u64(),
        value.c[2].as_canonical_u64(),
    ]
}

fn push_digest(message: &mut ChainMessage, digest: &Digest) {
    for word in digest_words(digest) {
        message.push_u64(word);
    }
}

fn push_point(message: &mut ChainMessage, point: &[Fp3]) {
    message.push_u64(point.len() as u64);
    for value in point {
        message.push_fp3(value);
    }
}

fn push_oracle_ref(message: &mut ChainMessage, oracle: &LinOracle) {
    push_digest(message, &oracle.root);
    push_digest(message, &oracle.start);
    message.push_u64(oracle.count);
}

fn push_oracle_message(message: &mut ChainMessage, oracle: &LinOracle) {
    push_oracle_ref(message, oracle);
    message.push_u64(oracle.values.len() as u64);
    for value in &oracle.values {
        for word in fp3_words(value) {
            message.push_u64(word);
        }
    }
}

fn push_shape(message: &mut ChainMessage, shape: &LinShape) {
    for value in [
        shape.matrix_rows,
        shape.k_lo as u64,
        shape.k_col as u64,
        shape.table_rows,
        shape.table_columns,
    ] {
        message.push_u64(value);
    }
}

fn commit_oracle(label: &Digest, values: &[Fp3]) -> Result<LinOracle> {
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(Error::InvalidConfig);
    }
    let rows: Vec<Vec<u64>> = values.iter().map(|value| fp3_words(value).to_vec()).collect();
    let mut registry = OracleRegistry::new();
    let (oracle, openings) = commit_rows_with_openings(&mut registry, label, &rows)?;
    let root = registry.get(oracle).ok_or(Error::ProverStateMissing)?.clone();
    Ok(LinOracle {
        root: root.root,
        start: root.start,
        count: root.count,
        values: values.to_vec(),
        openings,
    })
}

fn verify_oracle(oracle: &LinOracle) -> Result<()> {
    if oracle.values.len() as u64 != oracle.count
        || oracle.openings.len() as u64 != oracle.count
        || !oracle.count.is_power_of_two()
    {
        return Err(Error::ShapeMismatch);
    }
    let mut registry = OracleRegistry::new();
    let id = registry.insert(oracle.root, oracle.start, oracle.count)?;
    let depth = oracle.count.trailing_zeros() as usize;
    for (position, opening) in oracle.openings.iter().enumerate() {
        let expected = fp3_words(
            oracle
                .values
                .get(position)
                .ok_or(Error::ShapeMismatch)?,
        );
        if opening.oracle != id
            || opening.index != position as u64
            || opening.values != expected
            || opening.path.len() != depth
        {
            return Err(Error::InvalidProof);
        }
        verify_opening(&registry, opening, oracle.count)?;
    }
    Ok(())
}

/// Builds the LIN table from an authenticated asset opening. The trusted anchor
/// is verified first; the opening is then checked for identity, row order and
/// width, and the table entries are derived from the same opening.
pub fn build_authenticated_lin_table(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
) -> Result<LinTable> {
    verify_model_anchor(
        public,
        trusted.anchor_id,
        trusted.model_id,
        trusted.parameters_digest,
        trusted.asset_specs,
    )?;
    let meta: &AssetMeta = public
        .assets
        .iter()
        .find(|asset| asset.path == asset_path)
        .ok_or(Error::AssetMismatch)?;
    if opening.asset_id != meta.asset_id
        || opening.start != meta.start
        || opening.root != meta.root
        || opening.row_count != meta.rows
        || opening.rows.len() != meta.rows as usize
    {
        return Err(Error::StatementMismatch);
    }
    for (index, row) in opening.rows.iter().enumerate() {
        if row.row != index as u64 || row.values.len() != meta.row_width {
            return Err(Error::ShapeMismatch);
        }
    }
    public.verify_opening(opening)?;

    shape.check()?;
    let matrix_rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let matrix_columns =
        usize::try_from(shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
    let table_rows = usize::try_from(shape.table_rows).map_err(|_| Error::ResourceLimit)?;
    let table_columns =
        usize::try_from(shape.table_columns).map_err(|_| Error::ResourceLimit)?;
    if meta.row_width < matrix_columns || opening.rows.len() < matrix_rows {
        return Err(Error::ShapeMismatch);
    }
    let factor = 1u64
        .checked_shl(shape.k_lo)
        .ok_or(Error::ResourceLimit)?;
    let mut rows = vec![vec![Fp3::ZERO; table_columns]; table_rows];
    for row in 0..matrix_rows {
        let source = opening.rows.get(row).ok_or(Error::IncompleteArtifact)?;
        for column in 0..matrix_columns {
            let value = *source.values.get(column).ok_or(Error::ShapeMismatch)?;
            let table_row = (column as u64) >> shape.k_lo;
            let place = (column as u64) & (factor - 1);
            let entry = place
                .checked_add((row as u64) << shape.k_lo)
                .ok_or(Error::ResourceLimit)?;
            let cell = rows
                .get_mut(table_row as usize)
                .and_then(|values| values.get_mut(entry as usize))
                .ok_or(Error::ShapeMismatch)?;
            *cell = fp3_from_signed_word(value);
        }
    }
    LinTable::new(shape, rows)
}

fn begin_chain(binding: &BindingCandidate) -> Result<Chain> {
    Chain::begin(
        REQUEST_CHAIN_LABEL,
        &binding.parameters_digest,
        &binding.statement_bytes,
        &binding.record_bytes,
        None,
    )
}

fn begin_second_chain(
    binding: &BindingCandidate,
    first_query: &Digest,
    shared_seal: &Digest,
) -> Result<Chain> {
    let mut writer = CanonicalWriter::new();
    writer.put_u64(MESSAGE_SCHEMA_VERSION);
    writer.put_u64(STAGE_SECOND);
    writer.put_digest(first_query);
    writer.put_digest(shared_seal);
    let statement = writer.into_bytes();
    Chain::begin(
        FINAL_CHAIN_LABEL,
        &binding.parameters_digest,
        &statement,
        &binding.record_bytes,
        None,
    )
}

fn step_algebraic(chain: &mut Chain, message: ChainMessage) -> Result<()> {
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_first_commit(
    chain: &mut Chain,
    shape: &LinShape,
    w: &LinOracle,
    weights: &LinOracle,
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(MESSAGE_SCHEMA_VERSION);
    message.push_u64(STAGE_FIRST);
    message.push_u64(FIRST_PHASE_COMMIT);
    push_oracle_ref(&mut message, w);
    push_oracle_ref(&mut message, weights);
    push_shape(&mut message, shape);
    step_algebraic(chain, message)
}

fn absorb_first_claim(chain: &mut Chain, masks: &[LinMask], target: &Fp3) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(MESSAGE_SCHEMA_VERSION);
    message.push_u64(STAGE_FIRST);
    message.push_u64(FIRST_PHASE_CLAIM);
    message.push_u64(masks.len() as u64);
    for mask in masks {
        message.push_fp3(&mask.value);
        message.push_fp3(&mask.coefficient);
    }
    message.push_fp3(target);
    step_algebraic(chain, message)
}

fn absorb_shared_start(
    chain: &mut Chain,
    first_seal: &Digest,
    shape: &LinShape,
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(MESSAGE_SCHEMA_VERSION);
    message.push_u64(STAGE_SHARED);
    message.push_u64(SHARED_PHASE_START);
    push_digest(&mut message, first_seal);
    push_shape(&mut message, shape);
    step_algebraic(chain, message)
}

fn absorb_shared_commit(chain: &mut Chain, omega: &LinOracle, folded: &LinOracle) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(MESSAGE_SCHEMA_VERSION);
    message.push_u64(STAGE_SHARED);
    message.push_u64(SHARED_PHASE_OMEGA_FOLD);
    push_oracle_message(&mut message, omega);
    push_oracle_message(&mut message, folded);
    step_algebraic(chain, message)
}

#[allow(clippy::too_many_arguments)]
fn absorb_second(
    chain: &mut Chain,
    first_query: &Digest,
    shared_seal: &Digest,
    omega: &LinOracle,
    folded: &LinOracle,
    shared_point: &[Fp3],
    zeta: &[Fp3],
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(MESSAGE_SCHEMA_VERSION);
    message.push_u64(STAGE_SECOND);
    message.push_u64(SECOND_PHASE_FOLD);
    push_digest(&mut message, first_query);
    push_digest(&mut message, shared_seal);
    push_oracle_ref(&mut message, omega);
    push_oracle_ref(&mut message, folded);
    push_point(&mut message, shared_point);
    push_point(&mut message, zeta);
    step_algebraic(chain, message)
}

fn draw_point(chain: &mut Chain, count: u32) -> Result<Vec<Fp3>> {
    let count = usize::try_from(count).map_err(|_| Error::ResourceLimit)?;
    let mut point = Vec::new();
    point
        .try_reserve_exact(count)
        .map_err(|_| Error::ResourceLimit)?;
    for _ in 0..count {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

fn draw_psi(chain: &mut Chain, shape: &LinShape) -> Result<Vec<Fp3>> {
    draw_point(chain, shape.k_col)
}

fn draw_shared(chain: &mut Chain, shape: &LinShape) -> Result<Vec<Fp3>> {
    draw_point(chain, shape.k_col)
}

fn draw_zeta(chain: &mut Chain, shape: &LinShape) -> Result<Vec<Fp3>> {
    draw_point(chain, shape.k_row())
}

fn power_of_two_len(bits: u32) -> Result<usize> {
    let bits = usize::try_from(bits).map_err(|_| Error::ResourceLimit)?;
    1usize.checked_shl(bits as u32).ok_or(Error::ResourceLimit)
}

fn padded_weights(shape: &LinShape, weights: &[Fp3]) -> Result<Vec<Fp3>> {
    shape.check()?;
    if weights.len() != shape.matrix_rows as usize {
        return Err(Error::ShapeMismatch);
    }
    let slots = power_of_two_len(shape.k_row())?;
    let mut padded = vec![Fp3::ZERO; slots];
    padded[..weights.len()].copy_from_slice(weights);
    Ok(padded)
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

/// The folded message the fold relation reads: entry `j + 2^k_lo y` is the
/// table entry `(y, j + 2^k_lo r)` folded over the table rows `r` with the high
/// shared-point coordinates. Entries at or above `matrix_rows` stay zero
/// padding, matching the truncated row space.
fn folded_message(table: &LinTable, shared_point: &[Fp3]) -> Result<Vec<Fp3>> {
    let shape = table.shape;
    shape.check()?;
    if shared_point.len() != shape.k_col as usize {
        return Err(Error::ShapeMismatch);
    }
    let table_log = shape.table_rows.trailing_zeros();
    if shape.k_lo.checked_add(table_log) != Some(shape.k_col) {
        return Err(Error::InvalidConfig);
    }
    let places = power_of_two_len(shape.k_lo)?;
    let rows = power_of_two_len(shape.k_row())?;
    let length = rows.checked_mul(places).ok_or(Error::ResourceLimit)?;
    let mut folded = Vec::new();
    folded
        .try_reserve_exact(length)
        .map_err(|_| Error::ResourceLimit)?;
    folded.resize(length, Fp3::ZERO);
    let low = shape.k_lo as usize;
    let high = &shared_point[low..];
    for row in 0..shape.matrix_rows {
        for place in 0..places {
            let mut sum = Fp3::ZERO;
            for table_row in 0..shape.table_rows {
                let column = place as u64 + (table_row << shape.k_lo);
                let weight = eq_eval(high, table_row, table_log)?;
                sum = sum.add(&table.at(row, column)?.mul(&weight));
            }
            folded[place + row as usize * places] = sum;
        }
    }
    Ok(folded)
}

/// Full-value W control: recomputes `W = g^T M` from the table.
pub fn check_lin_w_control(table: &LinTable, weights: &[Fp3], w_values: &[Fp3]) -> Result<bool> {
    table.shape.check()?;
    if weights.len() != table.shape.matrix_rows as usize
        || w_values.len() as u64 != table.shape.matrix_columns()?
    {
        return Err(Error::ShapeMismatch);
    }
    Ok(table.w(weights)? == w_values)
}

/// Full-value Omega control: recomputes `Omega = M eq(psi, .)` from the table.
pub fn check_lin_omega_control(
    table: &LinTable,
    shared_point: &[Fp3],
    omega_values: &[Fp3],
) -> Result<bool> {
    table.shape.check()?;
    if shared_point.len() != table.shape.k_col as usize
        || omega_values.len() != power_of_two_len(table.shape.k_row())?
    {
        return Err(Error::ShapeMismatch);
    }
    Ok(table.omega(shared_point)? == omega_values)
}

/// Combined full-value control over W and Omega.
pub fn check_lin_control(
    table: &LinTable,
    weights: &[Fp3],
    shared_point: &[Fp3],
    w_values: &[Fp3],
    omega_values: &[Fp3],
) -> Result<LinControlChecks> {
    Ok(LinControlChecks {
        w_matches: check_lin_w_control(table, weights, w_values)?,
        omega_matches: check_lin_omega_control(table, shared_point, omega_values)?,
    })
}

/// Prover path. The W and weight commitments are fixed before `psi` is drawn;
/// `Omega` and the folded message are fixed after the shared point and before
/// `zeta` is drawn. The final chain binds the shared seal into the first
/// chain's first query.
#[allow(clippy::too_many_arguments)]
pub fn prove_lin_request(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
    weights: &[Fp3],
    masks: &[LinMask],
    w_values: &[Fp3],
) -> Result<LinRequestMessages> {
    let table = build_authenticated_lin_table(public, trusted, asset_path, opening, shape)?;
    prove_lin_request_with_table(&table, binding, shape, weights, masks, w_values)
}

/// Prover path over an already authenticated table. The caller must build the
/// table with `build_authenticated_lin_table`; this function does not
/// re-authenticate it.
pub(crate) fn prove_lin_request_with_table(
    table: &LinTable,
    binding: &BindingCandidate,
    shape: LinShape,
    weights: &[Fp3],
    masks: &[LinMask],
    w_values: &[Fp3],
) -> Result<LinRequestMessages> {
    if table.shape != shape {
        return Err(Error::ShapeMismatch);
    }
    shape.check()?;
    if w_values.len() as u64 != shape.matrix_columns()? {
        return Err(Error::ShapeMismatch);
    }
    let padded = padded_weights(&shape, weights)?;
    let w = commit_oracle(&W_LABEL, w_values)?;
    let weights_oracle = commit_oracle(&WEIGHTS_LABEL, &padded)?;

    let mut first_chain = begin_chain(binding)?;
    let first_query = *first_chain.value();
    if first_query != binding.first_query {
        return Err(Error::StatementMismatch);
    }
    absorb_first_commit(&mut first_chain, &shape, &w, &weights_oracle)?;
    let psi = draw_psi(&mut first_chain, &shape)?;

    let mut target = eval(w_values, &psi)?;
    for mask in masks {
        target = target.add(&mask.value.mul(&mask.coefficient));
    }
    absorb_first_claim(&mut first_chain, masks, &target)?;
    let first_seal = *first_chain.value();

    let mut shared_chain = Chain::resume(SHARED_CHAIN_LABEL, first_seal);
    absorb_shared_start(&mut shared_chain, &first_seal, &shape)?;
    let shared_point = draw_shared(&mut shared_chain, &shape)?;
    if shared_point == psi {
        return Err(Error::InvalidProof);
    }
    let omega_values = table.omega(&shared_point)?;
    let omega = commit_oracle(&OMEGA_LABEL, &omega_values)?;
    let folded_values = folded_message(table, &shared_point)?;
    let folded = commit_oracle(&FOLDED_LABEL, &folded_values)?;
    absorb_shared_commit(&mut shared_chain, &omega, &folded)?;
    let zeta = draw_zeta(&mut shared_chain, &shape)?;
    let shared_seal = *shared_chain.value();

    let mut second_chain = begin_second_chain(binding, &first_query, &shared_seal)?;
    absorb_second(
        &mut second_chain,
        &first_query,
        &shared_seal,
        &omega,
        &folded,
        &shared_point,
        &zeta,
    )?;
    let seal = *second_chain.value();

    Ok(LinRequestMessages {
        first: LinFirstMessage {
            w,
            weights: weights_oracle,
            psi,
            masks: masks.to_vec(),
            target,
        },
        shared: LinSharedMessage {
            first_seal,
            point: shared_point,
            omega,
            folded,
            zeta,
        },
        second: LinSecondMessage {
            first_query,
            shared_seal,
            seal,
        },
    })
}

/// Verifier path. It reconstructs the three chains in the prover's order,
/// compares every recorded commitment reference, point and seal, verifies the
/// four oracle openings, runs the Eval, Dot and Fold relations, and runs the
/// full-value W and Omega controls.
#[allow(clippy::too_many_arguments)]
pub fn verify_lin_request(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
    messages: &LinRequestMessages,
) -> Result<LinRequestOutcome> {
    let table = build_authenticated_lin_table(public, trusted, asset_path, opening, shape)?;
    verify_lin_request_with_table(&table, binding, shape, messages)
}

/// Verifier path over an already authenticated table. The caller must build
/// the table with `build_authenticated_lin_table`; this function does not
/// re-authenticate it.
pub(crate) fn verify_lin_request_with_table(
    table: &LinTable,
    binding: &BindingCandidate,
    shape: LinShape,
    messages: &LinRequestMessages,
) -> Result<LinRequestOutcome> {
    if table.shape != shape {
        return Err(Error::ShapeMismatch);
    }
    shape.check()?;
    let matrix_rows = usize::try_from(shape.matrix_rows).map_err(|_| Error::ResourceLimit)?;
    let matrix_columns =
        usize::try_from(shape.matrix_columns()?).map_err(|_| Error::ResourceLimit)?;
    let omega_len = power_of_two_len(shape.k_row())?;
    let folded_bits = shape
        .k_lo
        .checked_add(shape.k_row())
        .ok_or(Error::ResourceLimit)?;
    let folded_len = power_of_two_len(folded_bits)?;
    if messages.first.w.values.len() != matrix_columns
        || messages.first.weights.values.len() != omega_len
        || messages.first.psi.len() != shape.k_col as usize
        || messages.shared.point.len() != shape.k_col as usize
        || messages.shared.omega.values.len() != omega_len
        || messages.shared.folded.values.len() != folded_len
        || messages.shared.zeta.len() != shape.k_row() as usize
    {
        return Err(Error::ShapeMismatch);
    }
    verify_oracle(&messages.first.w)?;
    verify_oracle(&messages.first.weights)?;
    verify_oracle(&messages.shared.omega)?;
    verify_oracle(&messages.shared.folded)?;

    let mut first_chain = begin_chain(binding)?;
    let first_query = *first_chain.value();
    if first_query != binding.first_query || messages.second.first_query != first_query {
        return Err(Error::InvalidProof);
    }
    absorb_first_commit(
        &mut first_chain,
        &shape,
        &messages.first.w,
        &messages.first.weights,
    )?;
    let psi = draw_psi(&mut first_chain, &shape)?;
    if psi != messages.first.psi {
        return Err(Error::InvalidProof);
    }
    absorb_first_claim(&mut first_chain, &messages.first.masks, &messages.first.target)?;
    let first_seal = *first_chain.value();
    if first_seal != messages.shared.first_seal {
        return Err(Error::InvalidProof);
    }

    let weights = messages
        .first
        .weights
        .values
        .get(..matrix_rows)
        .ok_or(Error::ShapeMismatch)?;
    let claim = LinClaim {
        weights: weights.to_vec(),
        psi: psi.clone(),
        masks: messages.first.masks.clone(),
        target: messages.first.target,
    };
    claim.check(&shape)?;
    let eval_holds = lin_eval_holds_with(table, &claim, &messages.first.w.values)?;
    if !eval_holds {
        return Err(Error::InvalidProof);
    }
    let control_w = check_lin_w_control(table, weights, &messages.first.w.values)?;
    if !control_w {
        return Err(Error::InvalidProof);
    }

    let mut shared_chain = Chain::resume(SHARED_CHAIN_LABEL, first_seal);
    absorb_shared_start(&mut shared_chain, &first_seal, &shape)?;
    let shared_point = draw_shared(&mut shared_chain, &shape)?;
    if shared_point != messages.shared.point {
        return Err(Error::InvalidProof);
    }
    if shared_point == psi {
        return Err(Error::InvalidProof);
    }
    absorb_shared_commit(
        &mut shared_chain,
        &messages.shared.omega,
        &messages.shared.folded,
    )?;
    let zeta = draw_zeta(&mut shared_chain, &shape)?;
    if zeta != messages.shared.zeta {
        return Err(Error::InvalidProof);
    }
    let shared_seal = *shared_chain.value();
    if shared_seal != messages.second.shared_seal {
        return Err(Error::InvalidProof);
    }

    let dot_holds = lin_dot_holds_with_w(
        table,
        &claim,
        &shared_point,
        &messages.shared.omega.values,
        &messages.first.w.values,
    )?;
    if !dot_holds {
        return Err(Error::InvalidProof);
    }
    let control_omega =
        check_lin_omega_control(table, &shared_point, &messages.shared.omega.values)?;
    if !control_omega {
        return Err(Error::InvalidProof);
    }

    let mut second_chain = begin_second_chain(binding, &first_query, &shared_seal)?;
    absorb_second(
        &mut second_chain,
        &first_query,
        &shared_seal,
        &messages.shared.omega,
        &messages.shared.folded,
        &shared_point,
        &zeta,
    )?;
    let seal = *second_chain.value();
    if seal != messages.second.seal {
        return Err(Error::InvalidProof);
    }

    let psi_lo = &shared_point[..shape.k_lo as usize];
    let fold_holds = lin_fold_holds(
        shape,
        &messages.shared.omega.values,
        psi_lo,
        &zeta,
        &messages.shared.folded.values,
    )?;
    if !fold_holds {
        return Err(Error::InvalidProof);
    }

    Ok(LinRequestOutcome {
        first_query,
        first_seal,
        shared_seal,
        seal,
        psi,
        shared: shared_point,
        zeta,
        eval_holds,
        dot_holds,
        fold_holds,
        control_w,
        control_omega,
    })
}

/// Verifier path over the canonical message encoding. Decoding rejects trailing
/// bytes and stage blocks in the wrong order before the transcript is rebuilt.
#[allow(clippy::too_many_arguments)]
pub fn verify_lin_request_bytes(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
    bytes: &[u8],
) -> Result<LinRequestOutcome> {
    let messages = LinRequestMessages::decode(bytes)?;
    verify_lin_request(
        public, trusted, binding, asset_path, opening, shape, &messages,
    )
}

impl LinRequestMessages {
    /// Canonical positional encoding: one version word followed by the three
    /// length-prefixed stage blocks.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(MESSAGE_SCHEMA_VERSION);
        writer.put_bytes(&self.first.encode());
        writer.put_bytes(&self.shared.encode());
        writer.put_bytes(&self.second.encode());
        writer.into_bytes()
    }

    /// Decodes exactly one canonical message set and rejects trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let limits = DecodeLimits {
            max_bytes: bytes.len(),
            max_elements: bytes.len(),
        };
        let mut reader = CanonicalReader::new(bytes, limits);
        if reader.get_u64()? != MESSAGE_SCHEMA_VERSION {
            return Err(Error::InvalidEncoding);
        }
        let first = LinFirstMessage::decode(reader.get_bytes()?)?;
        let shared = LinSharedMessage::decode(reader.get_bytes()?)?;
        let second = LinSecondMessage::decode(reader.get_bytes()?)?;
        reader.finish()?;
        Ok(Self {
            first,
            shared,
            second,
        })
    }
}

fn stage_limits(bytes: &[u8]) -> DecodeLimits {
    DecodeLimits {
        max_bytes: bytes.len(),
        max_elements: bytes.len(),
    }
}

fn encode_point(writer: &mut CanonicalWriter, point: &[Fp3]) {
    writer.put_u64(point.len() as u64);
    for value in point {
        writer.put_fp3(value);
    }
}

fn decode_point(reader: &mut CanonicalReader<'_>) -> Result<Vec<Fp3>> {
    let count = reader.get_u64()?;
    let count = usize::try_from(count).map_err(|_| Error::InvalidEncoding)?;
    if count > reader.remaining().len() / FP3_BYTES {
        return Err(Error::InvalidEncoding);
    }
    let mut point = Vec::with_capacity(count);
    for _ in 0..count {
        point.push(reader.get_fp3()?);
    }
    Ok(point)
}

fn encode_masks(writer: &mut CanonicalWriter, masks: &[LinMask]) {
    writer.put_u64(masks.len() as u64);
    for mask in masks {
        writer.put_fp3(&mask.value);
        writer.put_fp3(&mask.coefficient);
    }
}

fn decode_masks(reader: &mut CanonicalReader<'_>) -> Result<Vec<LinMask>> {
    let count = reader.get_u64()?;
    let count = usize::try_from(count).map_err(|_| Error::InvalidEncoding)?;
    if count > reader.remaining().len() / (2 * FP3_BYTES) {
        return Err(Error::InvalidEncoding);
    }
    let mut masks = Vec::with_capacity(count);
    for _ in 0..count {
        masks.push(LinMask {
            value: reader.get_fp3()?,
            coefficient: reader.get_fp3()?,
        });
    }
    Ok(masks)
}

fn encode_oracle(writer: &mut CanonicalWriter, oracle: &LinOracle) {
    writer.put_digest(&oracle.root);
    writer.put_digest(&oracle.start);
    writer.put_u64(oracle.count);
    writer.put_u64(oracle.values.len() as u64);
    for value in &oracle.values {
        writer.put_fp3(value);
    }
    writer.put_u64(oracle.openings.len() as u64);
    for opening in &oracle.openings {
        writer.put_u64(opening.oracle.0);
        writer.put_u64(opening.index);
        writer.put_u64(opening.values.len() as u64);
        for value in &opening.values {
            writer.put_u64(*value);
        }
        writer.put_u64(opening.path.len() as u64);
        for digest in &opening.path {
            writer.put_digest(digest);
        }
    }
}

fn decode_u64_vec(reader: &mut CanonicalReader<'_>) -> Result<Vec<u64>> {
    let count = reader.get_u64()?;
    let count = usize::try_from(count).map_err(|_| Error::InvalidEncoding)?;
    if count > reader.remaining().len() / 8 {
        return Err(Error::InvalidEncoding);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.get_u64()?);
    }
    Ok(values)
}

fn decode_digest_vec(reader: &mut CanonicalReader<'_>) -> Result<Vec<Digest>> {
    let count = reader.get_u64()?;
    let count = usize::try_from(count).map_err(|_| Error::InvalidEncoding)?;
    if count > reader.remaining().len() / 32 {
        return Err(Error::InvalidEncoding);
    }
    let mut digests = Vec::with_capacity(count);
    for _ in 0..count {
        digests.push(reader.get_digest()?);
    }
    Ok(digests)
}

fn decode_oracle(reader: &mut CanonicalReader<'_>) -> Result<LinOracle> {
    let root = reader.get_digest()?;
    let start = reader.get_digest()?;
    let count = reader.get_u64()?;
    let values = decode_point(reader)?;
    let opening_count = reader.get_u64()?;
    let opening_count = usize::try_from(opening_count).map_err(|_| Error::InvalidEncoding)?;
    if opening_count > reader.remaining().len() / 8 {
        return Err(Error::InvalidEncoding);
    }
    let mut openings = Vec::with_capacity(opening_count);
    for _ in 0..opening_count {
        let oracle = OracleId(reader.get_u64()?);
        let index = reader.get_u64()?;
        let values = decode_u64_vec(reader)?;
        let path = decode_digest_vec(reader)?;
        openings.push(OracleOpening {
            oracle,
            index,
            values,
            path,
        });
    }
    if values.len() as u64 != count || openings.len() as u64 != count {
        return Err(Error::InvalidEncoding);
    }
    Ok(LinOracle {
        root,
        start,
        count,
        values,
        openings,
    })
}

impl LinFirstMessage {
    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(STAGE_FIRST);
        encode_oracle(&mut writer, &self.w);
        encode_oracle(&mut writer, &self.weights);
        encode_point(&mut writer, &self.psi);
        encode_masks(&mut writer, &self.masks);
        writer.put_fp3(&self.target);
        writer.into_bytes()
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, stage_limits(bytes));
        if reader.get_u64()? != STAGE_FIRST {
            return Err(Error::InvalidEncoding);
        }
        let w = decode_oracle(&mut reader)?;
        let weights = decode_oracle(&mut reader)?;
        let psi = decode_point(&mut reader)?;
        let masks = decode_masks(&mut reader)?;
        let target = reader.get_fp3()?;
        reader.finish()?;
        Ok(Self {
            w,
            weights,
            psi,
            masks,
            target,
        })
    }
}

impl LinSharedMessage {
    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(STAGE_SHARED);
        writer.put_digest(&self.first_seal);
        encode_point(&mut writer, &self.point);
        encode_oracle(&mut writer, &self.omega);
        encode_oracle(&mut writer, &self.folded);
        encode_point(&mut writer, &self.zeta);
        writer.into_bytes()
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, stage_limits(bytes));
        if reader.get_u64()? != STAGE_SHARED {
            return Err(Error::InvalidEncoding);
        }
        let first_seal = reader.get_digest()?;
        let point = decode_point(&mut reader)?;
        let omega = decode_oracle(&mut reader)?;
        let folded = decode_oracle(&mut reader)?;
        let zeta = decode_point(&mut reader)?;
        reader.finish()?;
        Ok(Self {
            first_seal,
            point,
            omega,
            folded,
            zeta,
        })
    }
}

impl LinSecondMessage {
    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(STAGE_SECOND);
        writer.put_digest(&self.first_query);
        writer.put_digest(&self.shared_seal);
        writer.put_digest(&self.seal);
        writer.into_bytes()
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = CanonicalReader::new(bytes, stage_limits(bytes));
        if reader.get_u64()? != STAGE_SECOND {
            return Err(Error::InvalidEncoding);
        }
        let first_query = reader.get_digest()?;
        let shared_seal = reader.get_digest()?;
        let seal = reader.get_digest()?;
        reader.finish()?;
        Ok(Self {
            first_query,
            shared_seal,
            seal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::proof_header::PROOF_VERSION_WORD;
    use zkie_engine::pcs::request::binding::{assemble, BindingParameters, BindingStatement};
    use zkie_engine::pcs::request::params::RequestParams;

    use crate::proof::model_commitment::{
        commit_assets, AssetStore, ModelAssetSpec, ModelCommitmentPublic,
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-lin-request-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    struct Fixture {
        root: PathBuf,
        public: ModelCommitmentPublic,
        opening: AssetOpening,
        specs: Vec<ModelAssetSpec>,
        anchor_id: Digest,
        model_id: Digest,
        parameters_digest: Digest,
        binding: BindingCandidate,
        shape: LinShape,
        weights: Vec<Fp3>,
        masks: Vec<LinMask>,
        w_values: Vec<Fp3>,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn binding(public: &ModelCommitmentPublic) -> BindingCandidate {
        let request_parameters = RequestParams::default_with_code(4, 4, 0).unwrap();
        let plan_digest = Digest([0x33; 32]);
        let protocol_identity = Digest([0x44; 32]);
        let parameters = BindingParameters::with_plan_parts(
            PROOF_VERSION_WORD,
            protocol_identity,
            &request_parameters,
            &[plan_digest],
        )
        .unwrap();
        let statement = BindingStatement::new(
            PROOF_VERSION_WORD,
            protocol_identity,
            b"case".to_vec(),
            plan_digest,
            public.anchor.anchor_id,
        )
        .unwrap();
        assemble(&parameters, &statement, None).unwrap()
    }

    fn fixture() -> Fixture {
        let root = temp_dir();
        fs::write(
            root.join("table_i32.bin"),
            [1i32, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: "table_i32.bin".into(),
            element_count: 8,
            row_width: 4,
            encoding: "i32-le".into(),
        }];
        let commitment =
            commit_assets(&root, Digest([1; 32]), Digest([2; 32]), &specs).unwrap();
        let public = ModelCommitmentPublic::from_commitment(&commitment);
        let opening = AssetStore::open(&commitment)
            .unwrap()
            .open_all()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let anchor_id = public.anchor.anchor_id;
        let model_id = Digest([1; 32]);
        let parameters_digest = Digest([2; 32]);
        let shape = LinShape::new(2, 1, 2, 4).unwrap();
        let binding = binding(&public);
        let trusted = TrustedModel {
            anchor_id,
            model_id,
            parameters_digest,
            asset_specs: &specs,
        };
        let table = build_authenticated_lin_table(
            &public,
            &trusted,
            "table_i32.bin",
            &opening,
            shape,
        )
        .unwrap();
        let weights = vec![Fp3::from_u64(5), Fp3::from_u64(7)];
        let masks = vec![LinMask {
            value: Fp3::from_u64(17),
            coefficient: Fp3::from_u64(3),
        }];
        let w_values = table.w(&weights).unwrap();
        Fixture {
            root,
            public,
            opening,
            specs,
            anchor_id,
            model_id,
            parameters_digest,
            binding,
            shape,
            weights,
            masks,
            w_values,
        }
    }

    fn trusted(fixture: &Fixture) -> TrustedModel<'_> {
        TrustedModel {
            anchor_id: fixture.anchor_id,
            model_id: fixture.model_id,
            parameters_digest: fixture.parameters_digest,
            asset_specs: &fixture.specs,
        }
    }

    fn prove(fixture: &Fixture) -> LinRequestMessages {
        let trusted = trusted(fixture);
        prove_lin_request(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            &fixture.weights,
            &fixture.masks,
            &fixture.w_values,
        )
        .unwrap()
    }

    fn verify(fixture: &Fixture, messages: &LinRequestMessages) -> Result<LinRequestOutcome> {
        let trusted = trusted(fixture);
        verify_lin_request(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            messages,
        )
    }

    fn verify_bytes(fixture: &Fixture, bytes: &[u8]) -> Result<LinRequestOutcome> {
        let trusted = trusted(fixture);
        verify_lin_request_bytes(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            bytes,
        )
    }

    fn table(fixture: &Fixture) -> LinTable {
        let trusted = trusted(fixture);
        build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
        )
        .unwrap()
    }

    fn add_one(value: &Fp3) -> Fp3 {
        value.add(&Fp3::from_u64(1))
    }

    #[test]
    fn request_prover_and_verifier_agree_on_all_three_stages() {
        let fixture = fixture();
        let messages = prove(&fixture);
        let outcome = verify(&fixture, &messages).unwrap();
        let table = table(&fixture);

        assert_eq!(messages.first.psi, outcome.psi);
        assert_eq!(messages.shared.point, outcome.shared);
        assert_eq!(messages.shared.zeta, outcome.zeta);
        assert_eq!(messages.shared.first_seal, outcome.first_seal);
        assert_eq!(messages.second.shared_seal, outcome.shared_seal);
        assert_eq!(messages.second.seal, outcome.seal);
        assert_eq!(messages.second.first_query, outcome.first_query);
        assert_eq!(outcome.first_query, fixture.binding.first_query);
        assert_ne!(outcome.psi, outcome.shared);
        assert!(outcome.eval_holds && outcome.dot_holds && outcome.fold_holds);
        assert!(outcome.control_w && outcome.control_omega);

        assert_eq!(messages.first.w.values, fixture.w_values);
        assert_eq!(messages.first.w.values, table.w(&fixture.weights).unwrap());
        assert_eq!(
            messages.shared.omega.values,
            table.omega(&messages.shared.point).unwrap()
        );
        assert_eq!(
            messages.shared.folded.values,
            folded_message(&table, &messages.shared.point).unwrap()
        );
        let controls = check_lin_control(
            &table,
            &fixture.weights,
            &messages.shared.point,
            &messages.first.w.values,
            &messages.shared.omega.values,
        )
        .unwrap();
        assert!(controls.w_matches && controls.omega_matches);
    }

    #[test]
    fn request_is_deterministic_for_the_same_inputs() {
        let fixture = fixture();
        assert_eq!(prove(&fixture), prove(&fixture));
    }

    #[test]
    fn canonical_encoding_round_trips_and_rejects_trailing_bytes() {
        let fixture = fixture();
        let messages = prove(&fixture);
        let bytes = messages.encode();
        let decoded = LinRequestMessages::decode(&bytes).unwrap();
        assert_eq!(decoded, messages);
        assert!(verify(&fixture, &decoded).is_ok());
        assert!(verify_bytes(&fixture, &bytes).is_ok());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(LinRequestMessages::decode(&trailing).is_err());
        assert!(verify_bytes(&fixture, &trailing).is_err());

        let mut short = bytes;
        short.pop();
        assert!(LinRequestMessages::decode(&short).is_err());
    }

    #[test]
    fn changed_w_is_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        let mut values = messages.first.w.values.clone();
        values[0] = add_one(&values[0]);
        messages.first.w = commit_oracle(&W_LABEL, &values).unwrap();
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.first.w.root.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_weights_are_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        let mut values = messages.first.weights.values.clone();
        values[0] = add_one(&values[0]);
        messages.first.weights = commit_oracle(&WEIGHTS_LABEL, &values).unwrap();
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.first.weights.root.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_omega_is_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        let mut values = messages.shared.omega.values.clone();
        values[0] = add_one(&values[0]);
        messages.shared.omega = commit_oracle(&OMEGA_LABEL, &values).unwrap();
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.shared.omega.root.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_folded_is_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        let mut values = messages.shared.folded.values.clone();
        values[0] = add_one(&values[0]);
        messages.shared.folded = commit_oracle(&FOLDED_LABEL, &values).unwrap();
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.shared.folded.root.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_points_are_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        messages.first.psi[0] = add_one(&messages.first.psi[0]);
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.shared.point[0] = add_one(&messages.shared.point[0]);
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.shared.zeta[0] = add_one(&messages.shared.zeta[0]);
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_seals_and_first_query_are_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        messages.shared.first_seal.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.second.shared_seal.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.second.seal.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.second.first_query.0[0] ^= 1;
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn changed_target_or_mask_is_rejected() {
        let fixture = fixture();

        let mut messages = prove(&fixture);
        messages.first.target = add_one(&messages.first.target);
        assert!(verify(&fixture, &messages).is_err());

        let mut messages = prove(&fixture);
        messages.first.masks[0].coefficient = add_one(&messages.first.masks[0].coefficient);
        assert!(verify(&fixture, &messages).is_err());
    }

    #[test]
    fn stage_order_is_rejected() {
        let fixture = fixture();
        let messages = prove(&fixture);
        let bytes = messages.encode();

        let limits = DecodeLimits::unlimited();
        let mut reader = CanonicalReader::new(&bytes, limits);
        assert_eq!(reader.get_u64().unwrap(), MESSAGE_SCHEMA_VERSION);
        let first = reader.get_bytes().unwrap().to_vec();
        let shared = reader.get_bytes().unwrap().to_vec();
        let second = reader.get_bytes().unwrap().to_vec();
        reader.finish().unwrap();

        let mut writer = CanonicalWriter::new();
        writer.put_u64(MESSAGE_SCHEMA_VERSION);
        writer.put_bytes(&shared);
        writer.put_bytes(&first);
        writer.put_bytes(&second);
        let reordered = writer.into_bytes();
        assert!(LinRequestMessages::decode(&reordered).is_err());
        assert!(verify_bytes(&fixture, &reordered).is_err());

        let mut swapped = prove(&fixture);
        core::mem::swap(&mut swapped.first.psi, &mut swapped.shared.point);
        assert!(verify(&fixture, &swapped).is_err());
    }

    #[test]
    fn table_opening_root_row_and_path_are_checked() {
        let fixture = fixture();
        let messages = prove(&fixture);

        let mut root = fixture.opening.clone();
        root.root.0[0] ^= 1;
        let trusted = trusted(&fixture);
        assert!(build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &root,
            fixture.shape
        )
        .is_err());

        let mut value = fixture.opening.clone();
        value.rows[0].values[0] ^= 1;
        assert!(build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &value,
            fixture.shape
        )
        .is_err());

        let mut path = fixture.opening.clone();
        path.rows[0].path.push(Digest([9; 32]));
        assert!(build_authenticated_lin_table(
            &fixture.public,
            &trusted,
            "table_i32.bin",
            &path,
            fixture.shape
        )
        .is_err());

        assert!(verify(&fixture, &messages).is_ok());
    }

    #[test]
    fn wrong_table_opening_is_rejected_through_the_request_path() {
        let fixture = fixture();
        let messages = prove(&fixture);
        let mut opening = fixture.opening.clone();
        opening.rows[1].values[0] ^= 1;
        let trusted = trusted(&fixture);
        assert!(verify_lin_request(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &opening,
            fixture.shape,
            &messages
        )
        .is_err());
    }
}
