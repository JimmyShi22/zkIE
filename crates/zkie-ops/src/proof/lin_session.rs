//! Isolated LIN claim session wiring for a candidate linear claim.
//!
//! This module demonstrates component-level wiring of the candidate LIN claim
//! into the existing session and binding primitives. It is not a proof
//! verifier, and its result must not be used to accept a proof. The
//! authenticated full-recompute isolation control from the selected-step
//! baseline remains in place and is run alongside this demonstration.
//!
//! The table `M` is built from an authenticated asset opening: the trusted
//! model anchor is verified first, then the opening is checked with the
//! existing Merkle opener, including asset id, root, row order and width. The
//! candidate Eval and Dot relations use the submitted, authenticated W vector
//! directly; they do not recompute W. The candidate fold relation is invoked
//! through the existing LIN helper. A separately named row-wise control
//! recomputes W and Omega for full-value comparison.
//!
//! The weights, claim point, target, W message, Omega message and folded rows
//! are self-reported values. W and Omega openings only show membership in the
//! commitments absorbed here; they do not show that those commitments are the
//! output of a trusted graph predecessor. The upstream graph source `x` is not
//! closed in this step. FOLDEQ authentication, commitment consistency,
//! proximity, hiding and single-shot semantics remain unconnected.
//!
//! FOLDEQ `draw_queries` was reviewed. It draws column positions and requires a
//! full FOLDEQ plan with a reader, a verified statement and runtime oracles;
//! this isolated claim has no such plan. The shared and later extension points
//! are therefore drawn with the existing chain draw interface, and no parallel
//! generator is introduced.
//!
//! The base field has `p < 2^64` and `Fp3` is a cubic extension, so
//! `|Fp3| = p^3 < 2^192`. This demonstration gives no soundness bound: a
//! multivariate nonzero polynomial detection bound needs its degree and the
//! challenge distribution, and Omega is chosen after the shared challenge, so
//! a bound fixed before that point does not apply. Only local equality checks
//! and the deterministic full-value control are run.
//!
//! The binding is assembled by the caller with the existing binding candidate.
//! The candidate protocol identity and request parameters are demo
//! placeholders, not real protocol values. Fixed test seeds, where used, are
//! test-only and do not seed a production session. The field order, messages
//! and challenge positions are candidate-internal and replaceable.

use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::binding::BindingCandidate;
use zkie_engine::pcs::request::chain::{Chain, KIND_ALGEBRAIC};
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::pcs::request::lin::{
    lin_dot_holds_with_w, lin_eval_holds_with, lin_fold_holds, LinClaim, LinShape, LinTable,
};
use zkie_engine::pcs::request::params::REQUEST_CHAIN_LABEL;

use crate::proof::binding::{
    commit_rows_with_openings, verify_opening, OracleOpening, OracleRegistry,
};
use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelCommitmentPublic,
};
use crate::proof::session_baseline::TrustedModel;

const SESSION_SCHEMA_VERSION: u64 = 1;
const STAGE_BIND_W: u64 = 1;
const STAGE_OMEGA_FOLD: u64 = 2;

const W_LABEL: Digest = Digest([0x71; 32]);
const OMEGA_LABEL: Digest = Digest([0x72; 32]);
const FOLDED_LABEL: Digest = Digest([0x73; 32]);

/// A self-reported committed vector. The openings prove membership in the
/// committed root; they do not authenticate the source of the values.
#[derive(Clone, Debug)]
pub struct LinOracle {
    pub root: Digest,
    pub start: Digest,
    pub count: u64,
    pub values: Vec<Fp3>,
    pub openings: Vec<OracleOpening>,
}

#[derive(Clone, Debug)]
pub struct LinMessages {
    pub w: LinOracle,
    pub omega: LinOracle,
    pub folded: LinOracle,
}

#[derive(Clone, Debug)]
pub struct LinProverOutput {
    pub shared_rho: Digest,
    pub shared: Vec<Fp3>,
    pub zeta_rho: Digest,
    pub zeta: Vec<Fp3>,
    pub messages: LinMessages,
}

#[derive(Clone, Debug)]
pub struct LinVerifierOutcome {
    pub shared_rho: Digest,
    pub shared: Vec<Fp3>,
    pub zeta_rho: Digest,
    pub zeta: Vec<Fp3>,
    pub shared_dimension: usize,
    pub eval_holds: bool,
    pub dot_holds: bool,
    pub fold_holds: bool,
    pub control_w: bool,
    pub control_omega: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinComponentChecks {
    pub eval_holds: bool,
    pub dot_holds: bool,
    pub fold_holds: bool,
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

fn push_digest(message: &mut zkie_engine::pcs::request::chain::ChainMessage, digest: &Digest) {
    for word in digest_words(digest) {
        message.push_u64(word);
    }
}

fn push_oracle_ref(
    message: &mut zkie_engine::pcs::request::chain::ChainMessage,
    oracle: &LinOracle,
) {
    push_digest(message, &oracle.root);
    push_digest(message, &oracle.start);
    message.push_u64(oracle.count);
}

fn push_oracle_message(
    message: &mut zkie_engine::pcs::request::chain::ChainMessage,
    oracle: &LinOracle,
) {
    push_oracle_ref(message, oracle);
    message.push_u64(oracle.values.len() as u64);
    for value in &oracle.values {
        for word in fp3_words(value) {
            message.push_u64(word);
        }
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

fn absorb_bind_w(
    chain: &mut Chain,
    shape: &LinShape,
    claim_digest: &Digest,
    w: &LinOracle,
) -> Result<()> {
    // Stage 1 binds the caller's existing binding candidate. Its protocol
    // identity and request parameters are demo placeholders, not real protocol
    // values; this is a demonstration session challenge.
    let mut message = chain.message();
    message.push_u64(SESSION_SCHEMA_VERSION);
    message.push_u64(STAGE_BIND_W);
    push_digest(&mut message, claim_digest);
    for value in [
        shape.matrix_rows,
        shape.k_lo as u64,
        shape.k_col as u64,
        shape.table_rows,
        shape.table_columns,
    ] {
        message.push_u64(value);
    }
    push_oracle_ref(&mut message, w);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_shared(chain: &mut Chain, shape: &LinShape) -> Result<Vec<Fp3>> {
    let mut point = Vec::with_capacity(shape.k_col as usize);
    for _ in 0..shape.k_col {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

fn absorb_omega_fold(
    chain: &mut Chain,
    omega: &LinOracle,
    folded: &LinOracle,
) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(SESSION_SCHEMA_VERSION);
    message.push_u64(STAGE_OMEGA_FOLD);
    push_oracle_message(&mut message, omega);
    push_oracle_message(&mut message, folded);
    let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn draw_zeta(chain: &mut Chain, shape: &LinShape) -> Result<Vec<Fp3>> {
    let mut point = Vec::with_capacity(shape.k_row() as usize);
    for _ in 0..shape.k_row() {
        point.push(chain.draw_e()?);
    }
    Ok(point)
}

/// Prover path. The W commitment is fixed before the shared challenge is
/// drawn; Omega and the folded message are fixed before the later challenge is
/// drawn. The caller supplies the self-reported W message; table values are
/// authenticated before use.
pub fn prove_lin_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
    claim: &LinClaim,
    w_values: &[Fp3],
) -> Result<LinProverOutput> {
    let table = build_authenticated_lin_table(public, trusted, asset_path, opening, shape)?;
    claim.check(&shape)?;
    if w_values.len() as u64 != shape.matrix_columns()? {
        return Err(Error::ShapeMismatch);
    }
    // Stage 1: fix the self-reported W commitment before the shared challenge.
    let w = commit_oracle(&W_LABEL, w_values)?;

    let mut chain = begin_chain(binding)?;
    absorb_bind_w(&mut chain, &shape, &claim.digest()?, &w)?;
    let shared_rho = *chain.value();
    // Stage 2: the shared challenge is a real extension-field point over the
    // column variables.
    let shared = draw_shared(&mut chain, &shape)?;

    // Stage 3: fix Omega and the folded message after the shared point but
    // before the later challenge used by the fold check.
    let omega_values = table.omega(&shared)?;
    let omega = commit_oracle(&OMEGA_LABEL, &omega_values)?;
    let folded_values = omega_values.clone();
    let folded = commit_oracle(&FOLDED_LABEL, &folded_values)?;
    absorb_omega_fold(&mut chain, &omega, &folded)?;
    let zeta_rho = *chain.value();
    let zeta = draw_zeta(&mut chain, &shape)?;

    Ok(LinProverOutput {
        shared_rho,
        shared,
        zeta_rho,
        zeta,
        messages: LinMessages { w, omega, folded },
    })
}

/// Component-layer checks. Eval and Dot use the submitted W values directly;
/// Fold uses the submitted Omega and folded values. These are relation checks,
/// not a soundness bound.
pub fn check_lin_components(
    table: &LinTable,
    claim: &LinClaim,
    shared: &[Fp3],
    zeta: &[Fp3],
    messages: &LinMessages,
) -> Result<LinComponentChecks> {
    claim.check(&table.shape)?;
    let eval_holds = lin_eval_holds_with(table, claim, &messages.w.values)?;
    let dot_holds = lin_dot_holds_with_w(
        table,
        claim,
        shared,
        &messages.omega.values,
        &messages.w.values,
    )?;
    let psi_lo = &claim.psi[..table.shape.k_lo as usize];
    let fold_holds = lin_fold_holds(
        table.shape,
        &messages.omega.values,
        psi_lo,
        zeta,
        &messages.folded.values,
    )?;
    Ok(LinComponentChecks {
        eval_holds,
        dot_holds,
        fold_holds,
    })
}

/// Full-value control layer. It recomputes W and Omega from the authenticated
/// table and compares every submitted value. This is the deterministic
/// full-protection comparison and is not a proof-verifier acceptance.
pub fn check_lin_control(
    table: &LinTable,
    claim: &LinClaim,
    shared: &[Fp3],
    messages: &LinMessages,
) -> Result<LinControlChecks> {
    claim.check(&table.shape)?;
    let w_matches = table.w(&claim.weights)? == messages.w.values;
    let omega_matches = table.omega(shared)? == messages.omega.values;
    Ok(LinControlChecks {
        w_matches,
        omega_matches,
    })
}

/// Verifier path. It reconstructs the same binding and stage order, verifies
/// the three commitments, runs the candidate Eval, Dot and Fold checks, and
/// then runs the authenticated full-recompute isolation control for W and
/// Omega.
pub fn verify_lin_session(
    public: &ModelCommitmentPublic,
    trusted: &TrustedModel<'_>,
    binding: &BindingCandidate,
    asset_path: &str,
    opening: &AssetOpening,
    shape: LinShape,
    claim: &LinClaim,
    messages: &LinMessages,
) -> Result<LinVerifierOutcome> {
    let table = build_authenticated_lin_table(public, trusted, asset_path, opening, shape)?;
    claim.check(&shape)?;

    let mut chain = begin_chain(binding)?;
    absorb_bind_w(&mut chain, &shape, &claim.digest()?, &messages.w)?;
    let shared_rho = *chain.value();
    let shared = draw_shared(&mut chain, &shape)?;
    absorb_omega_fold(&mut chain, &messages.omega, &messages.folded)?;
    let zeta_rho = *chain.value();
    let zeta = draw_zeta(&mut chain, &shape)?;

    verify_oracle(&messages.w)?;
    verify_oracle(&messages.omega)?;
    verify_oracle(&messages.folded)?;

    let components = check_lin_components(&table, claim, &shared, &zeta, messages)?;
    if !components.eval_holds || !components.dot_holds || !components.fold_holds {
        return Err(Error::InvalidProof);
    }
    let control = check_lin_control(&table, claim, &shared, messages)?;
    if !control.w_matches || !control.omega_matches {
        return Err(Error::InvalidProof);
    }

    Ok(LinVerifierOutcome {
        shared_rho,
        shared,
        zeta_rho,
        zeta,
        shared_dimension: shape.k_col as usize,
        eval_holds: components.eval_holds,
        dot_holds: components.dot_holds,
        fold_holds: components.fold_holds,
        control_w: control.w_matches,
        control_omega: control.omega_matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zkie_core::proof_header::PROOF_VERSION_WORD;
    use zkie_engine::pcs::request::binding::{assemble, BindingParameters, BindingStatement};
    use zkie_engine::pcs::request::lin::eval;
    use zkie_engine::pcs::request::params::RequestParams;

    use crate::proof::model_commitment::{
        commit_assets, AssetStore, ModelAssetSpec, ModelCommitmentPublic,
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const TEST_SEED: u64 = 0x5eed;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zkie-lin-session-{}-{}",
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
        claim: LinClaim,
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
            [1i32, 2, 3, 4]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let specs = vec![ModelAssetSpec {
            asset_id: 0,
            path: "table_i32.bin".into(),
            element_count: 4,
            row_width: 2,
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
        let trusted = TrustedModel {
            anchor_id,
            model_id,
            parameters_digest,
            asset_specs: &specs,
        };
        let shape = LinShape::new(2, 0, 2, 2).unwrap();
        let binding = binding(&public);
        let table = build_authenticated_lin_table(
            &public,
            &trusted,
            "table_i32.bin",
            &opening,
            shape,
        )
        .unwrap();
        let weights = vec![Fp3::from_u64(5), Fp3::from_u64(7)];
        let w_values = table.w(&weights).unwrap();
        let psi = vec![Fp3::from_u64(TEST_SEED)];
        let target = eval(&w_values, &psi).unwrap();
        let claim = LinClaim {
            weights,
            psi,
            masks: Vec::new(),
            target,
        };
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
            claim,
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

    fn verify(fixture: &Fixture, messages: &LinMessages) -> Result<LinVerifierOutcome> {
        let trusted = trusted(fixture);
        verify_lin_session(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            &fixture.claim,
            messages,
        )
    }

    fn prove(fixture: &Fixture) -> LinProverOutput {
        let trusted = trusted(fixture);
        prove_lin_session(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            &fixture.claim,
            &fixture.w_values,
        )
        .unwrap()
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

    #[test]
    fn prover_and_verifier_agree_and_all_local_checks_hold() {
        let fixture = fixture();
        let prover = prove(&fixture);
        let verifier = verify(&fixture, &prover.messages).unwrap();
        assert_eq!(prover.shared_rho, verifier.shared_rho);
        assert_eq!(prover.shared, verifier.shared);
        assert_eq!(prover.zeta_rho, verifier.zeta_rho);
        assert_eq!(prover.zeta, verifier.zeta);
        assert_eq!(verifier.shared_dimension, fixture.shape.k_col as usize);
        assert!(verifier.eval_holds && verifier.dot_holds && verifier.fold_holds);
        assert!(verifier.control_w && verifier.control_omega);
        let table = table(&fixture);
        let components = check_lin_components(
            &table,
            &fixture.claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(components.eval_holds && components.dot_holds && components.fold_holds);
        let control =
            check_lin_control(&table, &fixture.claim, &prover.shared, &prover.messages).unwrap();
        assert!(control.w_matches && control.omega_matches);
    }

    #[test]
    fn fixed_test_seed_is_reproducible() {
        let fixture = fixture();
        let first = prove(&fixture);
        let second = prove(&fixture);
        assert_eq!(first.shared_rho, second.shared_rho);
        assert_eq!(first.shared, second.shared);
        assert_eq!(first.zeta_rho, second.zeta_rho);
        assert_eq!(first.zeta, second.zeta);
        assert_eq!(first.messages.w.root, second.messages.w.root);
        assert_eq!(first.messages.omega.root, second.messages.omega.root);
        assert_eq!(first.messages.folded.root, second.messages.folded.root);
    }

    #[test]
    fn wrong_submitted_w_fails_component_eval_and_dot_before_control() {
        let fixture = fixture();
        let mut prover = prove(&fixture);
        let mut values = prover.messages.w.values.clone();
        values[0] = Fp3::from_u64(999);
        prover.messages.w = commit_oracle(&W_LABEL, &values).unwrap();
        let table = table(&fixture);
        let components = check_lin_components(
            &table,
            &fixture.claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(!components.eval_holds);
        assert!(!components.dot_holds);
        let control =
            check_lin_control(&table, &fixture.claim, &prover.shared, &prover.messages).unwrap();
        assert!(!control.w_matches);
        assert!(verify(&fixture, &prover.messages).is_err());
    }

    #[test]
    fn wrong_omega_message_is_seen_by_component_and_control_layers() {
        let fixture = fixture();
        let mut prover = prove(&fixture);
        let mut values = prover.messages.omega.values.clone();
        values[0] = Fp3::from_u64(999);
        prover.messages.omega = commit_oracle(&OMEGA_LABEL, &values).unwrap();
        let table = table(&fixture);
        let components = check_lin_components(
            &table,
            &fixture.claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(!components.dot_holds || !components.fold_holds);
        let control =
            check_lin_control(&table, &fixture.claim, &prover.shared, &prover.messages).unwrap();
        assert!(!control.omega_matches);
        assert!(verify(&fixture, &prover.messages).is_err());
    }

    #[test]
    fn wrong_folded_message_fails_component_fold_but_not_control() {
        let fixture = fixture();
        let mut prover = prove(&fixture);
        let mut values = prover.messages.folded.values.clone();
        values[0] = Fp3::from_u64(999);
        prover.messages.folded = commit_oracle(&FOLDED_LABEL, &values).unwrap();
        let table = table(&fixture);
        let components = check_lin_components(
            &table,
            &fixture.claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(!components.fold_holds);
        let control =
            check_lin_control(&table, &fixture.claim, &prover.shared, &prover.messages).unwrap();
        assert!(control.w_matches && control.omega_matches);
        assert!(verify(&fixture, &prover.messages).is_err());
    }

    #[test]
    fn table_opening_root_row_and_path_are_checked() {
        let fixture = fixture();
        let prover = prove(&fixture);

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

        assert!(verify(&fixture, &prover.messages).is_ok());
    }

    #[test]
    fn claim_target_and_weights_tampering_is_rejected() {
        let fixture = fixture();
        let prover = prove(&fixture);
        let table = table(&fixture);
        let mut claim = fixture.claim.clone();
        claim.target = Fp3::from_u64(999);
        let components = check_lin_components(
            &table,
            &claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(!components.eval_holds);
        let control =
            check_lin_control(&table, &claim, &prover.shared, &prover.messages).unwrap();
        assert!(control.w_matches && control.omega_matches);
        let trusted = trusted(&fixture);
        assert!(verify_lin_session(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            &claim,
            &prover.messages
        )
        .is_err());

        let mut claim = fixture.claim.clone();
        claim.weights[0] = Fp3::from_u64(999);
        let components = check_lin_components(
            &table,
            &claim,
            &prover.shared,
            &prover.zeta,
            &prover.messages,
        )
        .unwrap();
        assert!(components.eval_holds);
        assert!(!components.dot_holds);
        let control =
            check_lin_control(&table, &claim, &prover.shared, &prover.messages).unwrap();
        assert!(!control.w_matches);
        assert!(control.omega_matches);
        assert!(verify_lin_session(
            &fixture.public,
            &trusted,
            &fixture.binding,
            "table_i32.bin",
            &fixture.opening,
            fixture.shape,
            &claim,
            &prover.messages
        )
        .is_err());
    }

    #[test]
    fn replaced_commitment_root_changes_the_challenge_or_rejects() {
        let fixture = fixture();
        let mut prover = prove(&fixture);
        prover.messages.w.root.0[0] ^= 1;
        assert!(verify(&fixture, &prover.messages).is_err());
    }

    #[test]
    fn message_field_order_is_bound() {
        let fixture = fixture();
        let mut prover = prove(&fixture);
        prover.messages.omega.values.swap(0, 1);
        assert!(verify(&fixture, &prover.messages).is_err());
    }
}

