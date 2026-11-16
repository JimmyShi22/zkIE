//! The consumer shared LATER stage's column plan and durable charge.
//!
//! The stage reads a table batch of shared columns per run. The one-time
//! columns and the declared later batch form the instance total; the durable
//! table-life count is charged one fixed batch per run and is never reset by a
//! later run. The charge is the same primitive the table key stage uses.

use zkie_core::error::{Error, Result};
use zkie_engine::pcs::request::field::E_DEGREE;
use zkie_engine::pcs::request::read::HidingCharge;

use crate::proof::consumer_request::ConsumerSharedPlan;
use crate::proof::hidden_send_session::{shared_column_charge, HiddenTableProver, ReadLedgerCell};

/// The fixed column geometry of one consumer shared run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerSharedColumns {
    pub one_time_columns: u64,
    pub later_columns: u64,
    pub declared_batches: u32,
    /// The columns one run draws: the one-time batch.
    pub column_draws: usize,
    /// The per-run draw limit; it equals the run's own draw count.
    pub column_budget: usize,
    /// The fixed batch size the durable table-life count reserves.
    pub batch_columns: u64,
    /// The instance total: one-time columns plus every declared later batch.
    pub max_columns: u64,
}

impl ConsumerSharedColumns {
    pub fn new(shared: &ConsumerSharedPlan) -> Result<Self> {
        if shared.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        // The approved later rule: one declared later batch whose columns are
        // one fixed batch, so the instance total is one-time plus that batch.
        // The row counts follow the later rule: a later goal cannot exceed the
        // one-time goal.
        if shared.declared_batches != 1
            || shared.later_columns != shared.one_time_columns
            || shared.m_later == 0
            || shared.m_later > shared.m_once
        {
            return Err(Error::InvalidConfig);
        }
        let max_columns = shared
            .one_time_columns
            .checked_add(
                shared
                    .later_columns
                    .checked_mul(u64::from(shared.declared_batches))
                    .ok_or(Error::ResourceLimit)?,
            )
            .ok_or(Error::ResourceLimit)?;
        if max_columns > shared.table_tail {
            return Err(Error::InvalidConfig);
        }
        let column_draws =
            usize::try_from(shared.one_time_columns).map_err(|_| Error::ResourceLimit)?;
        Ok(Self {
            one_time_columns: shared.one_time_columns,
            later_columns: shared.later_columns,
            declared_batches: shared.declared_batches,
            column_draws,
            column_budget: column_draws,
            batch_columns: shared.one_time_columns,
            max_columns,
        })
    }
}

/// The shared stage's column geometry and its verifier-side read ledger. The
/// ledger's maximum is the instance total; a run draws one batch.
#[derive(Clone, Debug)]
pub struct ConsumerSharedStage {
    pub columns: ConsumerSharedColumns,
    pub ledger: ReadLedgerCell,
}

impl ConsumerSharedStage {
    pub fn new(shared: &ConsumerSharedPlan) -> Result<Self> {
        let columns = ConsumerSharedColumns::new(shared)?;
        let ledger = ReadLedgerCell::new(columns.max_columns);
        Ok(Self { columns, ledger })
    }

    pub fn fault(&self) -> Option<&'static str> {
        if self.columns.column_draws == 0
            || self.columns.column_draws > self.columns.column_budget
            || self.columns.max_columns == 0
            || self.ledger.max_reads().ok() != Some(self.columns.max_columns)
        {
            return Some("the consumer shared column plan is not settled");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }
}

/// Charges one shared run: the durable table-life count reserves exactly one
/// fixed batch before any column is read, and the in-memory hiding state
/// begins the run's own column count. A plan whose batch differs from the
/// table's fixed batch is refused, and no charge is returned without a
/// durable reservation.
pub fn consumer_shared_charge(
    stage: &ConsumerSharedStage,
    table: &mut dyn HiddenTableProver,
) -> Result<HidingCharge> {
    stage.check()?;
    shared_column_charge(
        table,
        stage.columns.batch_columns,
        stage.columns.column_draws,
    )
}

// ---------------------------------------------------------------------------
// The shared stage: binding-root start, Gu/Omega/opening/proximity
// ---------------------------------------------------------------------------

use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::CodeParams;
use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
use zkie_engine::pcs::request::gu::GuShape;
use zkie_engine::pcs::request::runtime_tree::MaskTreePlan;
use zkie_engine::pcs::request::table::{IndexReader, IndexShape};

use crate::proof::hidden_send_session::{
    prove_shared_public, shared_start_chain_from_binding, verify_shared_public,
    HiddenSharedMessages, ProximityBaseOutcome, SharedStagePlan,
};
use crate::proof::lin_binding::{
    binding_root_from_path, LinBindingPath, LinBindingRoot, LinBindingShape, LinBindingTree,
};

/// The trusted public geometry of one consumer shared stage. Every field is
/// plan data: the folded table message and the `Omega` message share one code,
/// the mask tree hides the fresh mask, and the table row reader is the same
/// table the consumer request names. Nothing here comes from a proof message.
#[derive(Clone, Debug)]
pub struct ConsumerSharedGeometry {
    /// The fixed binding shape and configuration the first chain's seal is
    /// bound under. The root itself is a per-request value derived from the
    /// verified first seal and the proof's path; it is never a static field.
    pub binding_shape: LinBindingShape,
    pub binding_configuration: Digest,
    pub table_identity: Digest,
    pub table_configuration: Digest,
    pub table_shape: IndexShape,
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub columns: u64,
    pub code: CodeParams,
    pub mask_code: CodeParams,
    pub mask_tree: MaskTreePlan,
    pub fresh: u64,
    pub base_positions: usize,
    pub gu: GuShape,
    pub omega: OmegaShapePlan,
    pub column_work_bits: u8,
    /// The digest the shared chain starts from: fixed statement data.
    pub parameters: Digest,
    /// The fixed statement bytes mixed into the chain's first round.
    pub statement: Vec<u8>,
    /// The reader's public name record, mixed into the chain's first round.
    pub record: Vec<u8>,
}

impl ConsumerSharedGeometry {
    pub fn fault(&self) -> Option<&'static str> {
        if self.parameters == Digest::ZERO
            || self.table_identity == Digest::ZERO
            || self.table_configuration == Digest::ZERO
            || self.binding_shape.fault().is_some()
            || self.binding_configuration == Digest::ZERO
            || self.columns == 0
            || !self.columns.is_power_of_two()
            || self.k_lo.checked_add(self.table_shape.v) != Some(self.columns.trailing_zeros())
        {
            return Some("the consumer shared geometry is not settled");
        }
        if self.gu.fault().is_some()
            || zkie_engine::pcs::request::relation::gu_layout_fault(&self.gu).is_some()
            || self.omega.fault().is_some()
            || self.mask_tree.fault().is_some()
        {
            return Some("a consumer shared shape is not a group");
        }
        if self.gu.index != self.table_shape
            || self.gu.u.code != self.code
            || self.omega.u.code != self.code
            || self.gu.u.answers != 1
            || self.omega.u.answers != 1
            || self.gu.u.mask_rows != 0
            || self.omega.u.mask_rows != 0
            || !self.gu.u.main_rows.is_power_of_two()
            || !self.omega.u.main_rows.is_power_of_two()
            || self.gu.group != self.omega.group
            || self.omega.matrix_rows != self.matrix_rows
            || self.omega.k_lo != self.k_lo
        {
            return Some("the consumer shared members do not join one group");
        }
        if self.mask_code.log_n != self.code.log_n
            || self.mask_tree.log_n != self.mask_code.log_n
            || self.fresh == 0
            || self.base_positions == 0
            || self.base_positions as u64 > self.code.r
            || self.base_positions as u64 > self.mask_code.r
        {
            return Some("the consumer shared mask or BASE is not settled");
        }
        let mut trees = vec![
            self.gu.tree.tree_number,
            self.omega.tree.tree_number,
            self.mask_tree.tree_number,
        ];
        let count = trees.len();
        trees.sort_unstable();
        trees.dedup();
        if trees.len() != count || self.gu.oracle == self.omega.oracle {
            return Some("a consumer shared identifier is not distinct");
        }
        None
    }

    pub fn check(&self) -> Result<()> {
        self.fault().map_or(Ok(()), |_| Err(Error::InvalidConfig))
    }

    fn stage<'a>(&'a self, stage: &'a ConsumerSharedStage) -> SharedStagePlan<'a> {
        SharedStagePlan {
            gu: self.gu.clone(),
            omega: self.omega.clone(),
            code: self.code,
            mask_code: self.mask_code,
            mask_tree: self.mask_tree.clone(),
            fresh: self.fresh,
            base_positions: self.base_positions,
            column_draws: stage.columns.column_draws,
            column_budget: stage.columns.column_budget,
            column_work_bits: self.column_work_bits,
            ledger: &stage.ledger,
            k_lo: self.k_lo,
            matrix_rows: self.matrix_rows,
            columns: self.columns,
            table_v: self.table_shape.v,
        }
    }

    fn check_reader(&self, reader: &IndexReader, table: &dyn HiddenTableProver) -> Result<()> {
        if reader.shape != self.table_shape
            || reader.configuration != self.table_configuration
            || table.reader().root != reader.root
            || reader_table_identity(reader)? != self.table_identity
        {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }
}

/// Builds the per-request binding tree over the verified first seal: one
/// place, empty siblings, the fixed shape/configuration and the request's
/// salt. The root and the request's path are proof values, not plan fields.
pub fn consumer_binding_prove(
    first_seal: &Digest,
    shape: LinBindingShape,
    configuration: &Digest,
    salts: Vec<u8>,
) -> Result<(LinBindingPath, LinBindingRoot)> {
    if *first_seal == Digest::ZERO || salts.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut tree = LinBindingTree::new(shape, *configuration, salts)?;
    tree.place(0, first_seal)?;
    let root = tree.root()?;
    let path = tree.path(0)?;
    Ok((path, root))
}

/// Rebuilds the binding root from the verifier's own verified first seal and
/// the proof's path. A wrong seal, salt, place or depth leads to another root
/// and therefore another shared chain.
pub fn consumer_binding_root(
    first_seal: &Digest,
    shape: &LinBindingShape,
    configuration: &Digest,
    path: &LinBindingPath,
) -> Result<LinBindingRoot> {
    if *first_seal == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    let root = binding_root_from_path(shape, configuration, first_seal, path)?;
    if root.fault().is_some() {
        return Err(Error::InvalidProof);
    }
    Ok(root)
}

/// The fixed identifiers and tree geometry of one consumer shared stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerSharedIds {
    pub group: u64,
    pub gu_oracle: u64,
    pub omega_oracle: u64,
    pub gu_tree: u64,
    pub omega_tree: u64,
    pub mask_tree: u64,
    pub fresh: u64,
    pub base_positions: usize,
    pub column_work_bits: u8,
}

fn shared_domain(label: &[u8], parameters: &Digest) -> Digest {
    let mut writer = zkie_engine::primitives::codec::CanonicalWriter::new();
    writer.put_bytes(label);
    writer.put_digest(parameters);
    zkie_core::artifact::digest_bytes(&writer.into_bytes())
}

impl ConsumerSharedGeometry {
    /// Derives the shared stage geometry from the first plan and the real
    /// table. The Gu and Omega codes are the first stage's W code, so the own
    /// group later joins W and Omega under one main code; the mask code is the
    /// first group's mask code. The trees, the fresh mask and the BASE size are
    /// the fixed plan identifiers in `ids`; their run and configuration
    /// digests are derived from the stage parameters, never random.
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        first_plan: &crate::proof::consumer_first::ConsumerFirstPlan,
        table_identity: Digest,
        table_configuration: Digest,
        table_shape: IndexShape,
        binding_shape: LinBindingShape,
        binding_configuration: Digest,
        parameters: Digest,
        statement: Vec<u8>,
        record: Vec<u8>,
        ids: ConsumerSharedIds,
    ) -> Result<Self> {
        let code = first_plan.draft.w_shape.code;
        let group = first_plan.group.as_ref().ok_or(Error::InvalidConfig)?;
        let mask_code = group.shapes.first_mask;
        let columns = first_plan.draft.columns;
        let k_lo = first_plan.draft.k_lo;
        if columns == 0
            || !columns.is_power_of_two()
            || k_lo.checked_add(table_shape.v) != Some(columns.trailing_zeros())
        {
            return Err(Error::InvalidConfig);
        }
        let tree = |tree_number: u64, label: &[u8], run_label: &[u8]| {
            zkie_engine::pcs::request::runtime_tree::RuntimeTreePlan {
                tree_number,
                leaf_class: 2,
                log_n: code.log_n,
                log_n_zk: mask_code.log_n,
                parts: vec![zkie_engine::pcs::request::runtime_tree::RuntimeTreePart {
                    main_symbols: E_DEGREE as u64,
                    mask_symbols: 0,
                }],
                salt_bytes: 40,
                run: shared_domain(run_label, &parameters),
                configuration: shared_domain(label, &parameters),
            }
        };
        let oracle = zkie_engine::pcs::request::gu::GuOracleShape {
            code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        };
        let geometry = Self {
            binding_shape,
            binding_configuration,
            table_identity,
            table_configuration,
            table_shape,
            matrix_rows: first_plan.draft.matrix_rows,
            k_lo,
            columns,
            code,
            mask_code,
            mask_tree: MaskTreePlan {
                tree_number: ids.mask_tree,
                leaf_class: 4,
                log_n: mask_code.log_n,
                mask_rows: 1,
                salt_bytes: 40,
                run: shared_domain(b"consumer-shared-mask-run", &parameters),
                configuration: shared_domain(b"consumer-shared-mask-config", &parameters),
            },
            fresh: ids.fresh,
            base_positions: ids.base_positions,
            gu: GuShape {
                oracle: ids.gu_oracle,
                group: ids.group,
                tree: tree(
                    ids.gu_tree,
                    b"consumer-shared-gu-config",
                    b"consumer-shared-gu-run",
                ),
                part: 0,
                rows_high: true,
                u: oracle.clone(),
                index: table_shape,
            },
            omega: OmegaShapePlan {
                oracle: ids.omega_oracle,
                group: ids.group,
                tree: tree(
                    ids.omega_tree,
                    b"consumer-shared-omega-config",
                    b"consumer-shared-omega-run",
                ),
                u: oracle,
                rows_high: true,
                matrix_rows: first_plan.draft.matrix_rows,
                k_lo,
            },
            column_work_bits: ids.column_work_bits,
            parameters,
            statement,
            record,
        };
        geometry.check()?;
        Ok(geometry)
    }
}

/// The prover's shared stage messages: the Gu and Omega roots, seeds and
/// answers, the drawn table columns and the proximity transcript.
pub struct ConsumerSharedProof {
    pub messages: HiddenSharedMessages,
    /// Prover-side only: the committed Gu and Omega material the second stage
    /// consumes. It is never encoded into the proof.
    pub(crate) witness: crate::proof::hidden_send_session::SharedWitness,
    pub(crate) points: crate::proof::hidden_send_session::SharedPoints,
}

/// The verifier's state after the shared stage: the shared seal and the
/// points, relation claims and BASE outcome the second stage consumes.
#[derive(Clone, Debug)]
pub struct ConsumerSharedVerified {
    pub seal: Digest,
    pub binding_root: LinBindingRoot,
    pub gu_root: Digest,
    pub gu_seeds: Vec<zkie_engine::pcs::request::table::SampleSeed>,
    pub gu_answers: Vec<zkie_engine::pcs::request::field::Fp3>,
    pub omega_root: Digest,
    pub omega_seeds: Vec<zkie_engine::pcs::request::table::SampleSeed>,
    pub omega_answers: Vec<zkie_engine::pcs::request::field::Fp3>,
    pub psi: Vec<zkie_engine::pcs::request::field::Fp3>,
    pub zeta: Vec<zkie_engine::pcs::request::field::Fp3>,
    pub claims: Vec<zkie_engine::pcs::request::relation::GuRelationRow>,
    pub base: ProximityBaseOutcome,
}

/// Prover side of the consumer shared stage over the caller's binding root.
/// The chain starts from the verified binding object, the table keeper owns
/// the rows and columns, and the run charges the same fixed batch it draws.
pub fn prove_consumer_shared(
    geometry: &ConsumerSharedGeometry,
    stage: &ConsumerSharedStage,
    reader: &IndexReader,
    table: &mut dyn HiddenTableProver,
    seed: Digest,
    charge: &HidingCharge,
    binding_root: &LinBindingRoot,
) -> Result<ConsumerSharedProof> {
    geometry.check()?;
    stage.check()?;
    geometry.check_reader(reader, table)?;
    if seed == Digest::ZERO {
        return Err(Error::InvalidConfig);
    }
    if charge.columns() != stage.columns.column_draws || charge.table() != reader.root {
        return Err(Error::StatementMismatch);
    }
    if binding_root.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let chain = shared_start_chain_from_binding(
        &geometry.parameters,
        &geometry.statement,
        geometry.table_shape.v as u64,
        geometry.k_lo,
        &geometry.record,
        binding_root,
    )?;
    let stage_plan = geometry.stage(stage);
    let result = prove_shared_public(&stage_plan, reader, chain, table, &[], None, &seed, charge)?;
    Ok(ConsumerSharedProof {
        messages: result.messages,
        witness: result.witness,
        points: result.points,
    })
}

/// Verifier side of the consumer shared stage. It starts from the same
/// binding root, checks the shared proof and returns the shared seal with the
/// points and claims the second stage consumes.
pub fn verify_consumer_shared(
    geometry: &ConsumerSharedGeometry,
    stage: &ConsumerSharedStage,
    reader: &IndexReader,
    messages: &HiddenSharedMessages,
    binding_root: &LinBindingRoot,
) -> Result<ConsumerSharedVerified> {
    geometry.check()?;
    stage.check()?;
    if reader.shape != geometry.table_shape
        || reader.configuration != geometry.table_configuration
        || reader_table_identity(reader)? != geometry.table_identity
    {
        return Err(Error::StatementMismatch);
    }
    if binding_root.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let chain = shared_start_chain_from_binding(
        &geometry.parameters,
        &geometry.statement,
        geometry.table_shape.v as u64,
        geometry.k_lo,
        &geometry.record,
        binding_root,
    )?;
    let stage_plan = geometry.stage(stage);
    let (seal, claims, points, base) =
        verify_shared_public(&stage_plan, reader, chain, &[], None, messages)?;
    if seal != messages.seal {
        return Err(Error::InvalidProof);
    }
    Ok(ConsumerSharedVerified {
        seal,
        binding_root: *binding_root,
        gu_root: messages.gu_root,
        gu_seeds: messages.gu_seeds.clone(),
        gu_answers: messages.gu_answers.clone(),
        omega_root: messages.omega_root,
        omega_seeds: messages.omega_seeds.clone(),
        omega_answers: messages.omega_answers.clone(),
        psi: points.psi,
        zeta: points.zeta,
        claims,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use zkie_engine::pcs::request::read::DurableColumnBudget;

    fn shared(one_time: u64, later: u64, batches: u32, tail: u64) -> ConsumerSharedPlan {
        ConsumerSharedPlan {
            m_once: 8,
            m_later: 8,
            one_time_columns: one_time,
            later_columns: later,
            declared_batches: batches,
            table_tail: tail,
        }
    }

    #[test]
    fn the_columns_follow_the_approved_later_rule() {
        let stage = ConsumerSharedStage::new(&shared(2, 2, 1, 4)).unwrap();
        assert_eq!(stage.columns.column_draws, 2);
        assert_eq!(stage.columns.column_budget, 2);
        assert_eq!(stage.columns.batch_columns, 2);
        assert_eq!(stage.columns.max_columns, 4);
        assert_eq!(stage.ledger.max_reads().unwrap(), 4);
        assert!(stage.fault().is_none());
        // Another later batch count, another batch size, a later goal above
        // the one-time goal and an instance total above the table tail are
        // refused.
        assert!(ConsumerSharedStage::new(&shared(2, 2, 2, 8)).is_err());
        assert!(ConsumerSharedStage::new(&shared(2, 1, 1, 4)).is_err());
        assert!(ConsumerSharedStage::new(&shared(2, 2, 1, 3)).is_err());
        let mut plan = shared(2, 2, 1, 4);
        plan.m_later = 9;
        assert!(ConsumerSharedStage::new(&plan).is_err());
        let mut plan = shared(2, 2, 1, 4);
        plan.one_time_columns = 0;
        assert!(ConsumerSharedStage::new(&plan).is_err());
    }

    #[test]
    fn the_durable_budget_reserves_one_fixed_batch_and_never_resets() {
        let root = std::env::temp_dir().join(format!(
            "zkie-consumer-shared-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let count: PathBuf = root.join("count.bin");
        DurableColumnBudget::initialize(&root, count.clone()).unwrap();
        let budget = DurableColumnBudget::open(&root, count, 2, 4).unwrap();
        assert_eq!(budget.columns(), 2);
        assert_eq!(budget.count().unwrap(), 0);
        budget.reserve().unwrap();
        budget.reserve().unwrap();
        assert_eq!(budget.count().unwrap(), 4);
        // The table-life limit is reached: no further batch is admitted and
        // the count is not reset.
        assert!(budget.reserve().is_err());
        assert_eq!(budget.count().unwrap(), 4);
        // Opening the same count again continues from it.
        let again = DurableColumnBudget::open(&root, root.join("count.bin"), 2, 4).unwrap();
        assert_eq!(again.count().unwrap(), 4);
        assert!(again.reserve().is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    // ------------------------------------------------------------------
    // The shared stage over a committed table
    // ------------------------------------------------------------------

    use std::cell::RefCell;

    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_core::types::Digest;
    use zkie_engine::pcs::request::code::{build_code_plan, CodeParams};
    use zkie_engine::pcs::request::field::{fp_from_u64, Fp, Fp3, E_DEGREE};
    use zkie_engine::pcs::request::foldeq::{reader_table_identity, OmegaShapePlan};
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    use zkie_engine::pcs::request::read::HidingState;
    use zkie_engine::pcs::request::runtime_tree::{MaskTreePlan, RuntimeTreePart, RuntimeTreePlan};
    use zkie_engine::pcs::request::table::{
        commit_table_streaming, IndexReader, IndexShape, StreamingTable, TableDistances,
        TableRowSource, VecTableSource,
    };

    use crate::proof::hidden_send_session::HiddenTableProver;
    use crate::proof::lin_binding::{
        binding_root_from_path, LinBindingPath, LinBindingRoot, LinBindingShape, LinBindingTree,
    };

    const TABLE_CONFIGURATION: Digest = Digest([0x21; 32]);

    fn code() -> CodeParams {
        build_code_plan(4, 8, 1).unwrap()
    }

    fn table_shape() -> IndexShape {
        IndexShape {
            code: code(),
            v: 2,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        }
    }

    fn shared_plan() -> ConsumerSharedPlan {
        ConsumerSharedPlan {
            m_once: 8,
            m_later: 8,
            one_time_columns: 1,
            later_columns: 1,
            declared_batches: 1,
            table_tail: 4,
        }
    }

    struct Keeper {
        table: StreamingTable,
        source: RefCell<VecTableSource>,
        reader: IndexReader,
        hiding: HidingState,
        budget: Option<DurableColumnBudget>,
    }

    impl TableRowSource for Keeper {
        fn rows(&self) -> u64 {
            self.source.borrow().rows()
        }

        fn row(&mut self, index: u64, message: &mut [Fp], tail: &mut [Fp]) -> Result<()> {
            self.source.borrow_mut().row(index, message, tail)
        }
    }

    impl HiddenTableProver for Keeper {
        fn reader(&self) -> &IndexReader {
            &self.reader
        }

        fn open_columns(
            &mut self,
            positions: &[u64],
        ) -> Result<Vec<zkie_engine::pcs::request::table::ColumnOpening>> {
            let mut source = self.source.borrow_mut();
            self.table.open_columns(&mut *source, positions)
        }

        fn hiding_state(&mut self) -> Option<&mut HidingState> {
            Some(&mut self.hiding)
        }

        fn durable_column_budget(&mut self) -> Option<&mut DurableColumnBudget> {
            self.budget.as_mut()
        }
    }

    fn keeper() -> (IndexReader, Keeper) {
        let shape = table_shape();
        let mut message_rows = vec![vec![Fp::ZERO; 8]; 4];
        for row in 0..4 {
            for column in 0..4 {
                message_rows[column][row] = crate::proof::column_session::fp_from_signed_word(
                    (row as i64 * 4 + column as i64 - 8) as u64,
                );
            }
        }
        let tails = vec![vec![Fp::ZERO; 1]; 4];
        let mut source = VecTableSource::new(message_rows, tails).unwrap();
        let salts: Vec<Vec<u8>> = (0..shape.code.n())
            .map(|position| vec![(position as u8) ^ 0x5A; 40])
            .collect();
        let table = commit_table_streaming(
            &mut source,
            shape,
            TABLE_CONFIGURATION,
            &salts,
            64,
            u64::MAX,
        )
        .unwrap();
        let reader = table.reader.clone();
        let keeper = Keeper {
            table,
            source: RefCell::new(source),
            hiding: HidingState::new(reader.root, 1, reader.shape.code.r as u64),
            reader: reader.clone(),
            budget: None,
        };
        (reader, keeper)
    }

    fn geometry(reader: &IndexReader) -> ConsumerSharedGeometry {
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
        ConsumerSharedGeometry {
            binding_shape: LinBindingShape {
                log_leaves: 0,
                salt_bytes: 40,
            },
            binding_configuration: Digest([0x92; 32]),
            table_identity: reader_table_identity(reader).unwrap(),
            table_configuration: TABLE_CONFIGURATION,
            table_shape: table_shape(),
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
                tree: tree(400, 2 * E_DEGREE as u64, 0x73, 0x74),
                part: 0,
                rows_high: true,
                u: oracle(2),
                index: table_shape(),
            },
            omega: OmegaShapePlan {
                oracle: 301,
                group: 8,
                tree: tree(401, E_DEGREE as u64, 0x75, 0x76),
                u: oracle(1),
                rows_high: true,
                matrix_rows: 4,
                k_lo: 0,
            },
            column_work_bits: 0,
            parameters: Digest([0x81; 32]),
            statement: b"consumer shared".to_vec(),
            record: reader.name(),
        }
    }

    fn binding_root(geom: &ConsumerSharedGeometry) -> (LinBindingPath, LinBindingRoot) {
        let salts: Vec<u8> = (0..40).map(|i| (i as u8) ^ 0x5A).collect();
        consumer_binding_prove(
            &Digest([0xA1; 32]),
            geom.binding_shape,
            &geom.binding_configuration,
            salts,
        )
        .unwrap()
    }

    #[test]
    fn the_shared_stage_proves_and_verifies_from_the_binding_root() {
        let (reader, mut keeper) = keeper();
        let geom = geometry(&reader);
        let (_path, root) = binding_root(&geom);
        let stage = ConsumerSharedStage::new(&shared_plan()).unwrap();
        let charge = consumer_shared_charge(&stage, &mut keeper).unwrap();
        assert_eq!(charge.columns(), 1);
        assert_eq!(charge.table(), reader.root);
        let seed = Digest([7; 32]);
        let proof =
            prove_consumer_shared(&geom, &stage, &reader, &mut keeper, seed, &charge, &root)
                .unwrap();
        let check = ConsumerSharedStage::new(&shared_plan()).unwrap();
        let verified =
            verify_consumer_shared(&geom, &check, &reader, &proof.messages, &root).unwrap();
        assert_eq!(verified.binding_root, root);
        assert_eq!(verified.seal, proof.messages.seal);
        assert_eq!(verified.gu_root, proof.messages.gu_root);
        assert_eq!(verified.omega_root, proof.messages.omega_root);
        assert!(!verified.psi.is_empty());
        assert!(!verified.zeta.is_empty());
        assert!(!verified.claims.is_empty());
        // A changed Gu root, Omega root, column nonce or point is refused on a
        // fresh ledger.
        let check = |changed: &crate::proof::hidden_send_session::HiddenSharedMessages| {
            let stage = ConsumerSharedStage::new(&shared_plan()).unwrap();
            verify_consumer_shared(&geom, &stage, &reader, changed, &root).is_err()
        };
        let mut changed = proof.messages.clone();
        changed.gu_root = Digest([0x92; 32]);
        assert!(check(&changed));
        let mut changed = proof.messages.clone();
        changed.omega_root = Digest([0x93; 32]);
        assert!(check(&changed));
        let mut changed = proof.messages.clone();
        changed.column_nonce ^= 1;
        assert!(check(&changed));
        let mut changed = proof.messages;
        changed.zeta[0] = changed.zeta[0].add(&Fp3::from_base(fp_from_u64(1)));
        assert!(check(&changed));
    }

    #[test]
    fn the_binding_root_starts_the_shared_chain() {
        let (reader, mut keeper) = keeper();
        let geom = geometry(&reader);
        let (path, root) = binding_root(&geom);
        let stage = ConsumerSharedStage::new(&shared_plan()).unwrap();
        let charge = consumer_shared_charge(&stage, &mut keeper).unwrap();
        let proof = prove_consumer_shared(
            &geom,
            &stage,
            &reader,
            &mut keeper,
            Digest([7; 32]),
            &charge,
            &root,
        )
        .unwrap();
        // Another salt, another first seal or a changed path place leads to
        // another root and the proof no longer verifies.
        let other_salts: Vec<u8> = (0..40).map(|i| (i as u8) ^ 0xC3).collect();
        let (other_path, other_root) = consumer_binding_prove(
            &Digest([0xA1; 32]),
            geom.binding_shape,
            &geom.binding_configuration,
            other_salts,
        )
        .unwrap();
        assert_ne!(other_root, root);
        assert_ne!(other_path, path);
        let stage = ConsumerSharedStage::new(&shared_plan()).unwrap();
        assert!(
            verify_consumer_shared(&geom, &stage, &reader, &proof.messages, &other_root).is_err()
        );
        let mut wrong_place = path.clone();
        wrong_place.place = 1;
        assert!(consumer_binding_root(
            &Digest([0xA1; 32]),
            &geom.binding_shape,
            &geom.binding_configuration,
            &wrong_place,
        )
        .is_err());
        // The verifier rebuilds the same root from its own first seal.
        let rebuilt = consumer_binding_root(
            &Digest([0xA1; 32]),
            &geom.binding_shape,
            &geom.binding_configuration,
            &path,
        )
        .unwrap();
        assert_eq!(rebuilt, root);
        let stage = ConsumerSharedStage::new(&shared_plan()).unwrap();
        let verified =
            verify_consumer_shared(&geom, &stage, &reader, &proof.messages, &rebuilt).unwrap();
        assert_eq!(verified.seal, proof.messages.seal);
    }
}
