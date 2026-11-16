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
    verify_consumer_shared, ConsumerSharedGeometry, ConsumerSharedStage, ConsumerSharedVerified,
};
use crate::proof::hidden_send_session::{HiddenSharedMessages, ProximityBaseOutcome};
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
    // The shared stage must verify against the same table reader.
    let verified = verify_consumer_shared(geometry, shared_stage, reader, shared)?;
    let binding = ConsumerChainBinding::from_first_shared(first, &verified)?;
    let chain = consumer_second_chain(geometry, &binding)?;
    let second_seal = *chain.value();
    if second_seal == Digest::ZERO
        || first.first_seal == Digest::ZERO
        || verified.seal == Digest::ZERO
    {
        return Err(Error::InvalidProof);
    }
    Ok(ConsumerTotalAccepted {
        first_seal: first.first_seal,
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

    use crate::proof::lin_binding::LinBindingRoot;

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
            binding_root: LinBindingRoot {
                digest: Digest([0x91; 32]),
                places: 1,
            },
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
}
