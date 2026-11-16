//! The consumer second chain and total acceptance.
//!
//! The second chain starts from the verified first claim (the first seal, the
//! request W root, its point and Eval target) and the verified shared result
//! (the shared seal, the Omega root, the shared point and zeta). Its parameter
//! digest binds every one of those objects, so another first claim, another W,
//! another shared result or another table starts another chain. Total
//! acceptance runs the first stage (its own group included), the shared stage
//! (Gu, Omega, the column openings and the proximity test) and then the second
//! chain; it returns only after every one of them verified.

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::chain::Chain;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::proof::consumer_first::{
    verify_consumer_first, ConsumerFirstMessages, ConsumerFirstObligations, ConsumerFirstPlan,
};
use crate::proof::consumer_shared::{
    consumer_binding_root, verify_consumer_shared, ConsumerSharedGeometry, ConsumerSharedStage,
    ConsumerSharedVerified,
};
use crate::proof::hidden_send_session::{HiddenSharedMessages, ProximityBaseOutcome};
use crate::proof::lin_binding::LinBindingPath;
use zkie_engine::pcs::request::table::IndexReader;

/// Version of the second chain's parameter digest.
pub const CONSUMER_SECOND_VERSION: u64 = 1;
/// The second chain's label.
pub const CONSUMER_SECOND_CHAIN_LABEL: u64 = 131;

/// The verified objects the second chain starts from: the first claim's seal,
/// W root and Eval target, and the shared seal, Omega root, shared point and
/// zeta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerChainBinding {
    pub first_seal: Digest,
    pub w_root: Digest,
    pub psi: Vec<Fp3>,
    pub eval_target: Fp3,
    pub shared_seal: Digest,
    pub omega_root: Digest,
    pub shared_point: Vec<Fp3>,
    pub zeta: Vec<Fp3>,
}

impl ConsumerChainBinding {
    /// Reads the binding from the verified first and shared results. Both
    /// sides are verifier outputs; no prover message is read here.
    pub fn from_first_shared(
        first: &ConsumerFirstMessages,
        shared: &ConsumerSharedVerified,
    ) -> Result<Self> {
        let binding = Self {
            first_seal: first.first_seal,
            w_root: first.w_root,
            psi: first.psi.clone(),
            eval_target: first.eval_target,
            shared_seal: shared.seal,
            omega_root: shared.omega_root,
            shared_point: shared.psi.clone(),
            zeta: shared.zeta.clone(),
        };
        binding.check()?;
        Ok(binding)
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.first_seal == Digest::ZERO
            || self.w_root == Digest::ZERO
            || self.shared_seal == Digest::ZERO
            || self.omega_root == Digest::ZERO
            || self.psi.is_empty()
            || self.shared_point.is_empty()
            || self.zeta.is_empty()
        {
            return Some("the consumer second binding is not settled");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// Starts the second chain. The fixed statement and reader record are the
/// shared stage's; the parameter digest binds the geometry's parameters, the
/// table identity and configuration and every verified binding object.
pub fn consumer_second_chain(
    geometry: &ConsumerSharedGeometry,
    binding: &ConsumerChainBinding,
) -> Result<Chain> {
    geometry.check()?;
    binding.check()?;
    let mut writer = CanonicalWriter::new();
    writer.put_u64(CONSUMER_SECOND_VERSION);
    writer.put_digest(&geometry.parameters);
    writer.put_digest(&geometry.table_identity);
    writer.put_digest(&geometry.table_configuration);
    writer.put_digest(&binding.first_seal);
    writer.put_digest(&binding.w_root);
    writer.put_u64(binding.psi.len() as u64);
    for value in &binding.psi {
        writer.put_fp3(value);
    }
    writer.put_fp3(&binding.eval_target);
    writer.put_digest(&binding.shared_seal);
    writer.put_digest(&binding.omega_root);
    writer.put_u64(binding.shared_point.len() as u64);
    for value in &binding.shared_point {
        writer.put_fp3(value);
    }
    writer.put_u64(binding.zeta.len() as u64);
    for value in &binding.zeta {
        writer.put_fp3(value);
    }
    let parameters = digest_bytes(&writer.into_bytes());
    Chain::begin(
        CONSUMER_SECOND_CHAIN_LABEL,
        &parameters,
        &geometry.statement,
        &geometry.record,
        None,
    )
}

/// The accepted consumer chain: both verified stages, the second chain and the
/// obligations they executed. It exists only when every check passed.
#[derive(Clone, Debug)]
pub struct ConsumerTotalAccepted {
    pub first_seal: Digest,
    pub binding_root: crate::proof::lin_binding::LinBindingRoot,
    pub shared_seal: Digest,
    pub second_seal: Digest,
    pub eval_target: Fp3,
    pub w_root: Digest,
    pub omega_root: Digest,
    pub obligations: ConsumerFirstObligations,
    pub shared_claims: usize,
    pub shared_base: ProximityBaseOutcome,
}

/// Total acceptance: runs the first stage (the request's own proximity group
/// included), the shared stage (Gu, Omega, the drawn columns and the proximity
/// test) and the second chain over the same first claim, W and shared Omega.
/// Only after all three verify does it return the accepted record.
#[allow(clippy::too_many_arguments)]
pub fn consumer_total_acceptance(
    plan: &ConsumerFirstPlan,
    activation: &[Fp3],
    outputs: &[Vec<i64>],
    first: &ConsumerFirstMessages,
    geometry: &ConsumerSharedGeometry,
    shared_stage: &ConsumerSharedStage,
    reader: &IndexReader,
    shared: &HiddenSharedMessages,
    binding_path: &LinBindingPath,
    own_plan: &ConsumerOwnPlan,
    own: &ConsumerOwnProof,
) -> Result<ConsumerTotalAccepted> {
    if plan.group.is_none() {
        return Err(Error::InvalidConfig);
    }
    // The request's table and the shared table must be the same asset.
    if plan.draft.request.table.identity != geometry.table_identity
        || plan.draft.request.table.configuration != geometry.table_configuration
    {
        return Err(Error::StatementMismatch);
    }
    // The first stage, including its own group, must verify.
    let obligations = verify_consumer_first(plan, activation, outputs, first)?;
    // The shared chain starts from the root rebuilt from this request's own
    // verified first seal and the proof's binding path.
    let binding_root = consumer_binding_root(
        &first.first_seal,
        &geometry.binding_shape,
        &geometry.binding_configuration,
        binding_path,
    )?;
    // The shared stage must verify against the same table reader.
    let verified = verify_consumer_shared(geometry, shared_stage, reader, shared, &binding_root)?;
    if verified.binding_root != binding_root {
        return Err(Error::InvalidProof);
    }
    // The own plan must consume the same W and the same shared Omega.
    if own_plan.w_root != first.w_root
        || own_plan.omega_root != verified.omega_root
        || own_plan.code != geometry.code
        || own_plan.w_oracle != plan.draft.w_oracle
    {
        return Err(Error::StatementMismatch);
    }
    let binding = ConsumerChainBinding::from_first_shared(first, &verified)?;
    let chain = consumer_second_chain(geometry, &binding)?;
    let own_verified = verify_consumer_own(chain, own_plan, &verified, own)?;
    let second_seal = own_verified.seal;
    if second_seal == Digest::ZERO
        || first.first_seal == Digest::ZERO
        || verified.seal == Digest::ZERO
    {
        return Err(Error::InvalidProof);
    }
    Ok(ConsumerTotalAccepted {
        first_seal: first.first_seal,
        binding_root,
        shared_seal: verified.seal,
        second_seal,
        eval_target: first.eval_target,
        w_root: first.w_root,
        omega_root: verified.omega_root,
        obligations,
        shared_claims: verified.claims.len(),
        shared_base: verified.base,
    })
}

// ---------------------------------------------------------------------------
// The second chain's own group: W sampling, the shared Omega and the Dot row
// ---------------------------------------------------------------------------

use std::sync::Arc;

use zkie_engine::pcs::request::bat::BatRow;
use zkie_engine::pcs::request::code::{encode_row_fp3, CodeParams};
use zkie_engine::pcs::request::field::E_DEGREE;
use zkie_engine::pcs::request::foldeq::OmegaShapePlan;
use zkie_engine::pcs::request::gu::GuOracleShape;
use zkie_engine::pcs::request::relation::GuRelationRow;
use zkie_engine::pcs::request::relation::{oracle_sample_row, sample_cube_of};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTreePlan, RuntimePartRows, RuntimeTree, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::SampleSeed;

use crate::proof::hidden_send_session::{
    decode_proximity_messages, encode_proximity_messages, member_fold_shape, pending_row_weights,
    proximity_prove, proximity_verify, request_dot_row, ProximityMember, ProximityTreeSource,
    SharedWitness, TableLayout, WShape,
};
use crate::proof::output_closure::CommittedValueGroup;

pub(crate) const CONSUMER_SECOND_ANSWER_TAG: u64 = 132;
pub(crate) const CONSUMER_SECOND_W_ROOT_TAG: u64 = 133;

/// The second chain's own plan: W and the shared Omega share one main code;
/// the fold uses its own fresh mask tree. Every field is trusted plan data.
#[derive(Clone, Debug)]
pub(crate) struct ConsumerOwnPlan {
    pub(crate) code: CodeParams,
    pub(crate) mask_code: CodeParams,
    pub(crate) mask_tree: MaskTreePlan,
    pub(crate) w_oracle: u64,
    pub(crate) w_shape: WShape,
    pub(crate) w_root: Digest,
    pub(crate) omega: OmegaShapePlan,
    pub(crate) omega_root: Digest,
    pub(crate) group: u64,
    pub(crate) fresh: u64,
    pub(crate) base_positions: usize,
    pub(crate) columns: u64,
    pub(crate) matrix_rows: u64,
    pub(crate) weights: Vec<Fp3>,
}

impl ConsumerOwnPlan {
    pub(crate) fn fault(&self) -> Option<&'static str> {
        if self.code != self.w_shape.code
            || self.omega.u.code != self.code
            || self.mask_code.log_n != self.code.log_n
            || self.mask_tree.log_n != self.mask_code.log_n
            || self.mask_tree.fault().is_some()
            || self.w_root == Digest::ZERO
            || self.omega_root == Digest::ZERO
            || self.fresh == 0
            || self.base_positions == 0
            || self.base_positions as u64 > self.code.r
            || self.base_positions as u64 > self.mask_code.r
            || self.columns == 0
            || !self.columns.is_power_of_two()
            || self.weights.len() as u64 != self.matrix_rows
            || self.w_oracle == self.omega.oracle
        {
            return Some("the consumer own plan is not settled");
        }
        let mut trees = vec![
            self.w_shape.tree_number,
            self.omega.tree.tree_number,
            self.mask_tree.tree_number,
        ];
        let count = trees.len();
        trees.sort_unstable();
        trees.dedup();
        if trees.len() != count {
            return Some("a consumer own identifier is not distinct");
        }
        None
    }

    pub(crate) fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    /// Builds the own plan from the first plan and the shared result. The W
    /// shape and root are the first stage's; the Omega shape is the shared
    /// stage's; the fold uses its own fresh mask tree.
    pub(crate) fn from_first_shared(
        first_plan: &ConsumerFirstPlan,
        first: &ConsumerFirstMessages,
        geometry: &crate::proof::consumer_shared::ConsumerSharedGeometry,
        shared: &crate::proof::consumer_shared::ConsumerSharedVerified,
        mask_tree: MaskTreePlan,
        fresh: u64,
        base_positions: usize,
    ) -> Result<Self> {
        Self::from_first_omega(
            first_plan,
            first,
            geometry,
            shared.omega_root,
            mask_tree,
            fresh,
            base_positions,
        )
    }

    /// The same own plan from the shared message's Omega root. The verifier
    /// builds the plan before the shared stage verifies; the total acceptance
    /// then requires the verified shared root to equal this one.
    pub(crate) fn from_first_omega(
        first_plan: &ConsumerFirstPlan,
        first: &ConsumerFirstMessages,
        geometry: &crate::proof::consumer_shared::ConsumerSharedGeometry,
        omega_root: Digest,
        mask_tree: MaskTreePlan,
        fresh: u64,
        base_positions: usize,
    ) -> Result<Self> {
        if omega_root == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        let plan = Self {
            code: geometry.code,
            mask_code: geometry.mask_code,
            mask_tree,
            w_oracle: first_plan.draft.w_oracle,
            w_shape: first_plan.draft.w_shape.clone(),
            w_root: first.w_root,
            omega: geometry.omega.clone(),
            omega_root,
            group: first_plan.draft.w_group,
            fresh,
            base_positions,
            columns: first_plan.draft.columns,
            matrix_rows: first_plan.draft.matrix_rows,
            weights: pending_row_weights(&first_plan.member, first_plan.draft.matrix_rows)?,
        };
        plan.check()?;
        Ok(plan)
    }
}

/// The prover's own-group material: the committed W message and its tree, the
/// shared Omega witness (message, tails, salts and opening) and the evaluated
/// sample answers.
pub(crate) struct ConsumerOwnWitness<'a> {
    pub w: &'a CommittedValueGroup,
    pub w_tails: &'a [Vec<Fp3>],
    pub shared: &'a SharedWitness,
}

/// The second chain's own proof: the drawn W seeds, their answers and the
/// proximity transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConsumerOwnProof {
    pub w_seeds: Vec<SampleSeed>,
    pub w_answers: Vec<Fp3>,
    pub messages: Vec<u8>,
    pub seal: Digest,
}

fn draw_sample_seed(chain: &mut Chain, bits: u32) -> Result<SampleSeed> {
    let mut z = Vec::with_capacity(bits as usize);
    for _ in 0..bits {
        z.push(chain.draw_e()?);
    }
    let beta = chain.draw_e_nonzero()?;
    Ok(SampleSeed { z, beta })
}

fn push_digest(message: &mut zkie_engine::pcs::request::chain::ChainMessage, digest: &Digest) {
    for word in digest.0.chunks_exact(8) {
        message.push_u64(u64::from_le_bytes(
            word.try_into().expect("eight-byte word"),
        ));
    }
}

fn absorb_own_w_root(chain: &mut Chain, oracle: u64, root: &Digest) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_SECOND_VERSION);
    message.push_u64(CONSUMER_SECOND_W_ROOT_TAG);
    message.push_u64(oracle);
    push_digest(&mut message, root);
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

fn absorb_own_answers(chain: &mut Chain, answers: &[Fp3]) -> Result<()> {
    let mut message = chain.message();
    message.push_u64(CONSUMER_SECOND_VERSION);
    message.push_u64(CONSUMER_SECOND_ANSWER_TAG);
    message.push_u64(answers.len() as u64);
    for answer in answers {
        message.push_fp3(answer);
    }
    let prefix = message.prefix(zkie_engine::pcs::request::chain::KIND_ALGEBRAIC, 0)?;
    if !chain.step(&prefix, 0)? {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// The own group's members in the BAT block order: W and the shared Omega.
/// The prover supplies the committed material; the verifier carries the roots
/// and messages it already verified.
fn own_members(
    plan: &ConsumerOwnPlan,
    witness: Option<&ConsumerOwnWitness<'_>>,
) -> Result<Vec<ProximityMember>> {
    plan.check()?;
    let (w_message, w_tails) = match witness {
        Some(witness) => {
            let message = witness
                .w
                .messages
                .first()
                .ok_or(Error::ShapeMismatch)?
                .clone();
            (message, witness.w_tails.to_vec())
        }
        None => (Vec::new(), Vec::new()),
    };
    let (omega_message, omega_tails) = match witness {
        Some(witness) => {
            let ell = usize::try_from(plan.omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
            let rows = usize::try_from(plan.omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
            if witness.shared.omega_message.len() != ell * rows {
                return Err(Error::ShapeMismatch);
            }
            let mut message_rows = Vec::with_capacity(rows);
            for row in 0..rows {
                message_rows
                    .push(witness.shared.omega_message[row * ell..(row + 1) * ell].to_vec());
            }
            (message_rows, witness.shared.omega_tails.clone())
        }
        None => (Vec::new(), Vec::new()),
    };
    let mut members = vec![
        ProximityMember {
            bat_oracle: 0,
            source_oracle: plan.w_oracle,
            main_code: plan.code,
            mask_code: plan.mask_code,
            main_rows: plan.w_shape.main_rows,
            mask_rows: 0,
            answers: plan.w_shape.answers,
            root: plan.w_root,
            tree_plan: plan.w_shape.tree_plan(0)?,
            message: w_message,
            tails: w_tails,
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
        ProximityMember {
            bat_oracle: 0,
            source_oracle: plan.omega.oracle,
            main_code: plan.code,
            mask_code: plan.mask_code,
            main_rows: plan.omega.u.main_rows,
            mask_rows: 0,
            answers: plan.omega.u.answers,
            root: plan.omega_root,
            tree_plan: plan.omega.tree.clone(),
            message: omega_message,
            tails: omega_tails,
            mask_message: Vec::new(),
            mask_tails: Vec::new(),
        },
    ];
    members.sort_by_key(|member| {
        std::cmp::Reverse((member.main_rows, member.mask_rows, member.source_oracle))
    });
    for (index, member) in members.iter_mut().enumerate() {
        member.bat_oracle = index as u64;
    }
    Ok(members)
}

fn own_sample_rows(
    plan: &ConsumerOwnPlan,
    members: &[ProximityMember],
    shared: &ConsumerSharedVerified,
    w_seeds: &[SampleSeed],
    w_answers: &[Fp3],
) -> Result<Vec<Vec<GuRelationRow>>> {
    let mut out = Vec::with_capacity(members.len());
    for member in members {
        let rows = if member.source_oracle == plan.w_oracle {
            let mut rows = Vec::with_capacity(w_answers.len());
            for (answer, target) in w_answers.iter().enumerate() {
                rows.push(oracle_sample_row(
                    plan.w_oracle,
                    plan.group,
                    &plan.w_shape.oracle_shape(),
                    w_seeds,
                    answer as u32,
                    *target,
                )?);
            }
            rows
        } else if member.source_oracle == plan.omega.oracle {
            let mut rows = Vec::with_capacity(shared.omega_answers.len());
            for (answer, target) in shared.omega_answers.iter().enumerate() {
                rows.push(oracle_sample_row(
                    plan.omega.oracle,
                    plan.omega.group,
                    &plan.omega.u,
                    &shared.omega_seeds,
                    answer as u32,
                    *target,
                )?);
            }
            rows
        } else {
            return Err(Error::StatementMismatch);
        };
        if rows.len() != member.answers as usize {
            return Err(Error::ShapeMismatch);
        }
        out.push(rows);
    }
    Ok(out)
}

/// The own group's plain row: the Dot row tying the request W to the shared
/// Omega at the shared point, with the pending claim's public weights. The row
/// target is zero on both sides.
fn own_plain_rows(
    plan: &ConsumerOwnPlan,
    members: &[ProximityMember],
    shared: &ConsumerSharedVerified,
) -> Result<Vec<BatRow>> {
    let bat_oracle = |oracle: u64| -> Result<u64> {
        members
            .iter()
            .find(|member| member.source_oracle == oracle)
            .map(|member| member.bat_oracle)
            .ok_or(Error::StatementMismatch)
    };
    let w_bat = bat_oracle(plan.w_oracle)?;
    let omega_bat = bat_oracle(plan.omega.oracle)?;
    let k_col = plan.columns.trailing_zeros() as usize;
    let psi_star = shared
        .psi
        .get(..k_col)
        .ok_or(Error::ShapeMismatch)?
        .to_vec();
    Ok(vec![request_dot_row(
        TableLayout {
            main_rows: plan.omega.u.main_rows,
            ell: plan.omega.u.code.ell,
        },
        plan.matrix_rows,
        omega_bat,
        &plan.weights,
        TableLayout::of(&plan.w_shape),
        plan.columns,
        w_bat,
        &psi_star,
    )?])
}

fn own_trees(
    plan: &ConsumerOwnPlan,
    members: &[ProximityMember],
    witness: &ConsumerOwnWitness<'_>,
) -> Result<Vec<(u64, Arc<RuntimeTree>)>> {
    let mut trees = Vec::with_capacity(members.len());
    for member in members {
        if member.source_oracle == plan.w_oracle {
            trees.push((
                plan.w_oracle,
                Arc::clone(witness.w.trees.0.first().ok_or(Error::ShapeMismatch)?),
            ));
        } else if member.source_oracle == plan.omega.oracle {
            let mut encoded = Vec::new();
            let ell = usize::try_from(plan.omega.u.code.ell).map_err(|_| Error::ResourceLimit)?;
            let rows = usize::try_from(plan.omega.u.main_rows).map_err(|_| Error::ResourceLimit)?;
            for row in 0..rows {
                let message = &witness.shared.omega_message[row * ell..(row + 1) * ell];
                let tail = witness
                    .shared
                    .omega_tails
                    .get(row)
                    .ok_or(Error::ShapeMismatch)?;
                encoded.push(encode_row_fp3(&plan.omega.u.code, message, tail)?);
            }
            let tree = RuntimeTree::commit(
                plan.omega.tree.clone(),
                &[RuntimePartRows {
                    main: encoded,
                    mask: Vec::new(),
                    main_deg: E_DEGREE as u32,
                }],
                &witness.shared.omega_salts,
            )?;
            if *tree.root() != member.root {
                return Err(Error::StatementMismatch);
            }
            trees.push((plan.omega.oracle, Arc::new(tree)));
        } else {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(trees)
}

fn evaluate_relation_row(row: &GuRelationRow, main: &[Fp3], mask: &[Fp3]) -> Result<Fp3> {
    if row.main.len() != main.len() || row.mask.len() != mask.len() {
        return Err(Error::ShapeMismatch);
    }
    let mut value = Fp3::ZERO;
    for (coefficient, entry) in row.main.iter().zip(main) {
        value = value.add(&coefficient.mul(entry));
    }
    for (coefficient, entry) in row.mask.iter().zip(mask) {
        value = value.add(&coefficient.mul(entry));
    }
    Ok(value)
}

/// Prover side of the second chain's own group: draw W seeds after the first
/// claim, use the shared Omega seeds and answers, and run the proximity test
/// over W, Omega and the Dot row.
pub(crate) fn prove_consumer_own(
    mut chain: Chain,
    plan: &ConsumerOwnPlan,
    shared: &ConsumerSharedVerified,
    witness: &ConsumerOwnWitness<'_>,
    run_seed: Digest,
) -> Result<ConsumerOwnProof> {
    plan.check()?;
    if run_seed == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    absorb_own_w_root(&mut chain, plan.w_oracle, &plan.w_root)?;
    let bits = sample_cube_of(&plan.w_shape.oracle_shape())?.bits;
    let mut w_seeds = Vec::with_capacity(plan.w_shape.answers as usize);
    for _ in 0..plan.w_shape.answers {
        w_seeds.push(draw_sample_seed(&mut chain, bits)?);
    }
    let w_message = witness
        .w
        .messages
        .first()
        .and_then(|rows| rows.first())
        .ok_or(Error::ShapeMismatch)?
        .clone();
    let mut w_answers = Vec::with_capacity(plan.w_shape.answers as usize);
    for answer in 0..plan.w_shape.answers {
        let row = oracle_sample_row(
            plan.w_oracle,
            plan.group,
            &plan.w_shape.oracle_shape(),
            &w_seeds,
            answer,
            Fp3::ZERO,
        )?;
        w_answers.push(evaluate_relation_row(&row, &w_message, &[])?);
    }
    absorb_own_answers(&mut chain, &w_answers)?;
    let members = own_members(plan, Some(witness))?;
    let sample_rows = own_sample_rows(plan, &members, shared, &w_seeds, &w_answers)?;
    let plain_rows = own_plain_rows(plan, &members, shared)?;
    let fold = member_fold_shape(
        plan.code,
        plan.mask_code,
        &members,
        plan.mask_tree.leaf_class,
    )?;
    let trees = own_trees(plan, &members, witness)?;
    let proof = proximity_prove(
        chain,
        fold,
        members,
        ProximityTreeSource::Ready {
            trees,
            omega: Some(witness.shared.omega_opening.clone()),
        },
        &sample_rows,
        plain_rows,
        plan.fresh,
        plan.mask_tree,
        run_seed,
        plan.base_positions,
        None,
    )?;
    Ok(ConsumerOwnProof {
        w_seeds,
        w_answers,
        messages: encode_proximity_messages(&proof.messages)?,
        seal: proof.seal,
    })
}

/// The verifier's own-group result: the own seal, the W sample and the
/// relation rows it checked.
#[derive(Clone, Debug)]
pub(crate) struct ConsumerOwnVerified {
    pub seal: Digest,
    pub w_seeds: Vec<SampleSeed>,
    pub w_answers: Vec<Fp3>,
    pub claims: Vec<GuRelationRow>,
    pub base: ProximityBaseOutcome,
}

/// Verifier side of the second chain's own group. It draws the same W seeds,
/// takes the W answers from the proof, uses the shared Omega state and runs
/// the same proximity test.
pub(crate) fn verify_consumer_own(
    mut chain: Chain,
    plan: &ConsumerOwnPlan,
    shared: &ConsumerSharedVerified,
    proof: &ConsumerOwnProof,
) -> Result<ConsumerOwnVerified> {
    plan.check()?;
    absorb_own_w_root(&mut chain, plan.w_oracle, &plan.w_root)?;
    let bits = sample_cube_of(&plan.w_shape.oracle_shape())?.bits;
    let mut w_seeds = Vec::with_capacity(plan.w_shape.answers as usize);
    for _ in 0..plan.w_shape.answers {
        w_seeds.push(draw_sample_seed(&mut chain, bits)?);
    }
    if w_seeds != proof.w_seeds {
        return Err(Error::InvalidProof);
    }
    if proof.w_answers.len() != plan.w_shape.answers as usize {
        return Err(Error::ShapeMismatch);
    }
    absorb_own_answers(&mut chain, &proof.w_answers)?;
    let members = own_members(plan, None)?;
    let sample_rows = own_sample_rows(plan, &members, shared, &w_seeds, &proof.w_answers)?;
    let plain_rows = own_plain_rows(plan, &members, shared)?;
    let fold = member_fold_shape(
        plan.code,
        plan.mask_code,
        &members,
        plan.mask_tree.leaf_class,
    )?;
    let messages = decode_proximity_messages(None, &proof.messages)?;
    let verified = proximity_verify(
        chain,
        fold,
        &members,
        &sample_rows,
        plain_rows,
        &messages,
        plan.fresh,
        plan.mask_tree,
        plan.base_positions,
        None,
    )?;
    if verified.seal != proof.seal {
        return Err(Error::InvalidProof);
    }
    let claims = sample_rows.into_iter().flatten().collect::<Vec<_>>();
    Ok(ConsumerOwnVerified {
        seal: verified.seal,
        w_seeds,
        w_answers: proof.w_answers.clone(),
        claims,
        base: verified.base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::{fp_from_u64, E_DEGREE};
    use zkie_engine::pcs::request::foldeq::OmegaShapePlan;
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
    use zkie_engine::pcs::request::table::{IndexShape, TableDistances};

    use crate::proof::lin_binding::{LinBindingRoot, LinBindingShape};

    fn code() -> zkie_engine::pcs::request::code::CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn geometry_fixture() -> ConsumerSharedGeometry {
        let c = code();
        let tree =
            |tree_number: u64, main_symbols: u64, run: u8, configuration: u8| RuntimeTreePlan {
                tree_number,
                leaf_class: 2,
                log_n: c.log_n,
                log_n_zk: c.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols,
                    mask_symbols: 0,
                }],
                salt_bytes: 40,
                run: Digest([run; 32]),
                configuration: Digest([configuration; 32]),
            };
        let oracle = |main_rows: u64| GuOracleShape {
            code: c,
            main_rows,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        };
        let table_shape = IndexShape {
            code: c,
            v: 2,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        ConsumerSharedGeometry {
            binding_shape: LinBindingShape {
                log_leaves: 0,
                salt_bytes: 40,
            },
            binding_configuration: Digest([0x92; 32]),
            table_identity: Digest([0x12; 32]),
            table_configuration: Digest([0x13; 32]),
            table_shape,
            matrix_rows: 4,
            k_lo: 0,
            columns: 4,
            code: c,
            mask_code: c,
            mask_tree: MaskTreePlan {
                tree_number: 402,
                leaf_class: 4,
                log_n: c.log_n,
                mask_rows: 1,
                salt_bytes: 40,
                run: Digest([0x77; 32]),
                configuration: Digest([0x78; 32]),
            },
            fresh: 96,
            base_positions: 1,
            gu: GuShape {
                oracle: 300,
                group: 8,
                tree: tree(400, 6, 0x73, 0x74),
                part: 0,
                rows_high: true,
                u: oracle(2),
                index: table_shape,
            },
            omega: OmegaShapePlan {
                oracle: 301,
                group: 8,
                tree: tree(401, 3, 0x75, 0x76),
                u: oracle(1),
                rows_high: true,
                matrix_rows: 4,
                k_lo: 0,
            },
            column_work_bits: 0,
            parameters: Digest([0x81; 32]),
            statement: b"consumer shared".to_vec(),
            record: b"record".to_vec(),
        }
    }

    fn binding_fixture() -> ConsumerChainBinding {
        ConsumerChainBinding {
            first_seal: Digest([0xA1; 32]),
            w_root: Digest([0xA2; 32]),
            psi: vec![
                Fp3::from_base(fp_from_u64(1)),
                Fp3::from_base(fp_from_u64(2)),
            ],
            eval_target: Fp3::from_base(fp_from_u64(3)),
            shared_seal: Digest([0xA3; 32]),
            omega_root: Digest([0xA4; 32]),
            shared_point: vec![Fp3::from_base(fp_from_u64(4))],
            zeta: vec![Fp3::from_base(fp_from_u64(5))],
        }
    }

    #[test]
    fn the_second_chain_binds_both_stages_and_the_shared_points() {
        let geometry = geometry_fixture();
        let binding = binding_fixture();
        let chain = consumer_second_chain(&geometry, &binding).unwrap();
        let seal = *chain.value();
        assert_ne!(seal, Digest::ZERO);
        let again = consumer_second_chain(&geometry, &binding).unwrap();
        assert_eq!(*again.value(), seal);
        // Another first seal, W root, shared seal, Omega root or point starts
        // another chain.
        let mut changed = binding.clone();
        changed.first_seal = Digest([0xB1; 32]);
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        let mut changed = binding.clone();
        changed.w_root = Digest([0xB2; 32]);
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        let mut changed = binding.clone();
        changed.shared_seal = Digest([0xB3; 32]);
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        let mut changed = binding.clone();
        changed.omega_root = Digest([0xB4; 32]);
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        let mut changed = binding.clone();
        changed.psi[0] = changed.psi[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        let original = binding.clone();
        let mut changed = binding;
        changed.zeta[0] = changed.zeta[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert_ne!(
            *consumer_second_chain(&geometry, &changed).unwrap().value(),
            seal
        );
        // Another table identity or configuration also starts another chain.
        let original = binding_fixture();
        let mut other = geometry_fixture();
        other.table_identity = Digest([0xC1; 32]);
        assert_ne!(
            *consumer_second_chain(&other, &original).unwrap().value(),
            seal
        );
        let mut other = geometry_fixture();
        other.table_configuration = Digest([0xC2; 32]);
        assert_ne!(
            *consumer_second_chain(&other, &original).unwrap().value(),
            seal
        );
    }

    #[test]
    fn an_unsettled_binding_is_refused() {
        let geometry = geometry_fixture();
        let mut binding = binding_fixture();
        binding.first_seal = Digest::ZERO;
        assert!(consumer_second_chain(&geometry, &binding).is_err());
        let mut binding = binding_fixture();
        binding.psi.clear();
        assert!(consumer_second_chain(&geometry, &binding).is_err());
        let mut binding = binding_fixture();
        binding.zeta.clear();
        assert!(consumer_second_chain(&geometry, &binding).is_err());
    }

    // ------------------------------------------------------------------
    // The second chain's own group over committed W and Omega messages
    // ------------------------------------------------------------------

    use zkie_engine::pcs::request::chain::Chain;
    use zkie_engine::pcs::request::code::encode_row_fp3;
    use zkie_engine::pcs::request::relation::{oracle_sample_row, sample_cube_of};
    use zkie_engine::pcs::request::runtime_tree::{RuntimePartRows, RuntimeTree};
    use zkie_engine::pcs::request::table::SampleSeed;
    use zkie_engine::pcs::request::tree_book::TreeOpeningCount;

    use crate::proof::consumer_shared::ConsumerSharedVerified;
    use crate::proof::hidden_send_session::{ProximityBaseOutcome, SharedWitness, WShape};
    use crate::proof::output_closure::commit_value_group;

    const OWN_RUN: Digest = Digest([0x71; 32]);
    const OWN_CONFIGURATION: Digest = Digest([0x72; 32]);
    const OWN_OMEGA_RUN: Digest = Digest([0x73; 32]);
    const OWN_OMEGA_CONFIGURATION: Digest = Digest([0x74; 32]);

    fn fp(value: u64) -> Fp3 {
        Fp3::from_base(fp_from_u64(value))
    }

    fn own_chain() -> Chain {
        Chain::begin(
            CONSUMER_SECOND_CHAIN_LABEL,
            &Digest([0xD1; 32]),
            b"consumer own",
            b"record",
            None,
        )
        .unwrap()
    }

    fn own_code() -> zkie_engine::pcs::request::code::CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn own_mask_tree() -> MaskTreePlan {
        MaskTreePlan {
            tree_number: 950,
            leaf_class: 4,
            log_n: own_code().log_n,
            mask_rows: 1,
            salt_bytes: 40,
            run: OWN_RUN,
            configuration: OWN_CONFIGURATION,
        }
    }

    fn balance(coins: &[u8]) -> Vec<u8> {
        let mut salt = Vec::with_capacity(40);
        while salt.len() < 40 {
            salt.extend_from_slice(coins);
        }
        salt.truncate(40);
        salt
    }

    struct OwnFixture {
        plan: ConsumerOwnPlan,
        shared: ConsumerSharedVerified,
        committed: crate::proof::output_closure::CommittedValueGroup,
        w_tails: Vec<Vec<Fp3>>,
        shared_witness: SharedWitness,
    }

    fn own_fixture() -> OwnFixture {
        let code = own_code();
        let ell = usize::try_from(code.ell).unwrap();
        let w_shape = WShape::new(4, code, 1, 1, 40, 700, 1, OWN_RUN, OWN_CONFIGURATION).unwrap();
        let omega_u = GuOracleShape {
            code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        };
        let omega_tree = RuntimeTreePlan {
            tree_number: 701,
            leaf_class: 2,
            log_n: code.log_n,
            log_n_zk: code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: E_DEGREE as u64,
                mask_symbols: 0,
            }],
            salt_bytes: 40,
            run: OWN_OMEGA_RUN,
            configuration: OWN_OMEGA_CONFIGURATION,
        };
        // The shared Omega message: four weighted places and a zero tail.
        let mut omega_message = vec![Fp3::ZERO; ell];
        for (place, value) in omega_message.iter_mut().take(4).enumerate() {
            *value = fp((place + 1) as u64);
        }
        let omega_tail = vec![Fp3::ZERO];
        let omega_salts: Vec<Vec<u8>> = (0..code.n())
            .map(|position| balance(&[(position as u8) ^ 0x31]))
            .collect();
        let encoded = encode_row_fp3(&code, &omega_message, &omega_tail).unwrap();
        let omega_runtime = RuntimeTree::commit(
            omega_tree.clone(),
            &[RuntimePartRows {
                main: vec![encoded],
                mask: Vec::new(),
                main_deg: E_DEGREE as u32,
            }],
            &omega_salts,
        )
        .unwrap();
        let omega_root = *omega_runtime.root();
        let omega_opening = TreeOpeningCount::new(0, 701, omega_root, code.r).unwrap();
        // The Dot row holder: weights [1,2,3,4] over Omega [1,2,3,4] and W[0]
        // carries the same sum, so the row target is zero.
        let weights: Vec<Fp3> = (1..=4).map(fp).collect();
        let dot = weights
            .iter()
            .zip(omega_message.iter())
            .fold(Fp3::ZERO, |sum, (weight, value)| {
                sum.add(&weight.mul(value))
            });
        let w_values = vec![vec![dot, Fp3::ZERO, Fp3::ZERO, Fp3::ZERO]];
        let pads = vec![fp(9)];
        let w_tails = vec![vec![Fp3::ZERO]];
        let salts: Vec<Vec<Vec<u8>>> = vec![(0..code.n())
            .map(|position| balance(&[(position as u8) ^ 0x21]))
            .collect()];
        let committed =
            commit_value_group(&w_shape, 4, 50, 5, &w_values, &pads, &w_tails, &salts, 1).unwrap();
        let w_root = committed.roots[0];
        // The shared Omega sample seed and its answer.
        let bits = sample_cube_of(&omega_u).unwrap().bits;
        let omega_seed = SampleSeed {
            z: (0..bits).map(|index| fp(index as u64 + 1)).collect(),
            beta: fp(5),
        };
        let omega_row =
            oracle_sample_row(51, 7, &omega_u, &[omega_seed.clone()], 0, Fp3::ZERO).unwrap();
        let omega_answer = omega_row
            .main
            .iter()
            .zip(&omega_message)
            .fold(Fp3::ZERO, |sum, (coefficient, value)| {
                sum.add(&coefficient.mul(value))
            });
        let shared = ConsumerSharedVerified {
            seal: Digest([0xA3; 32]),
            binding_root: LinBindingRoot {
                digest: Digest([0x91; 32]),
                places: 1,
            },
            gu_root: Digest([0xA4; 32]),
            gu_seeds: Vec::new(),
            gu_answers: Vec::new(),
            omega_root,
            omega_seeds: vec![omega_seed],
            omega_answers: vec![omega_answer],
            psi: vec![Fp3::ZERO, Fp3::ZERO],
            zeta: vec![Fp3::ZERO, Fp3::ZERO],
            claims: Vec::new(),
            base: ProximityBaseOutcome::Stages {
                last_positions: Vec::new(),
            },
        };
        let plan = ConsumerOwnPlan {
            code,
            mask_code: code,
            mask_tree: own_mask_tree(),
            w_oracle: 50,
            w_shape,
            w_root,
            omega: zkie_engine::pcs::request::foldeq::OmegaShapePlan {
                oracle: 51,
                group: 7,
                tree: omega_tree,
                u: omega_u,
                rows_high: true,
                matrix_rows: 4,
                k_lo: 0,
            },
            omega_root,
            group: 7,
            fresh: 77,
            base_positions: 1,
            columns: 4,
            matrix_rows: 4,
            weights,
        };
        plan.check().unwrap();
        let shared_witness = SharedWitness {
            gu_message: Vec::new(),
            gu_tails: Vec::new(),
            omega_message,
            omega_tails: vec![omega_tail],
            omega_salts,
            omega_opening,
        };
        OwnFixture {
            plan,
            shared,
            committed,
            w_tails,
            shared_witness,
        }
    }

    #[test]
    fn the_second_own_group_samples_w_and_joins_the_shared_omega() {
        let fixture = own_fixture();
        let witness = ConsumerOwnWitness {
            w: &fixture.committed,
            w_tails: &fixture.w_tails,
            shared: &fixture.shared_witness,
        };
        let proof = prove_consumer_own(
            own_chain(),
            &fixture.plan,
            &fixture.shared,
            &witness,
            Digest([0xE1; 32]),
        )
        .unwrap();
        let verified =
            verify_consumer_own(own_chain(), &fixture.plan, &fixture.shared, &proof).unwrap();
        assert_eq!(verified.seal, proof.seal);
        assert_eq!(verified.w_seeds, proof.w_seeds);
        assert_eq!(verified.w_answers, proof.w_answers);
        assert!(!verified.claims.is_empty());
        // Another W answer, another shared Omega answer or another weight row
        // no longer verifies.
        let mut changed = proof.clone();
        changed.w_answers[0] = changed.w_answers[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert!(
            verify_consumer_own(own_chain(), &fixture.plan, &fixture.shared, &changed).is_err()
        );
        let mut changed = proof.clone();
        changed.w_seeds[0].z[0] = changed.w_seeds[0].z[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert!(
            verify_consumer_own(own_chain(), &fixture.plan, &fixture.shared, &changed).is_err()
        );
        let mut other_shared = fixture.shared.clone();
        other_shared.omega_answers[0] =
            other_shared.omega_answers[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert!(verify_consumer_own(own_chain(), &fixture.plan, &other_shared, &proof).is_err());
        let mut other_plan = fixture.plan.clone();
        other_plan.weights[0] = other_plan.weights[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert!(verify_consumer_own(own_chain(), &other_plan, &fixture.shared, &proof).is_err());
    }
}
