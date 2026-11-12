//! FOLDEQ session skeleton (foundation only).
//!
//! The verifier's plan holds the committed table, the fold shape, the runtime
//! mask-tree layout, the group registry and the budgets/counts; every number
//! comes from the verified statement/plan, not from a proof. With a G_u shape
//! the session orders the complete first query, the record seeds, the gamma
//! challenge, the received G_u root, the shared seed set drawn from that
//! session chain, the answer round (closed before any later challenge), the
//! group join and record rows, the column draws, the column rows and the FOLD
//! start; without one it keeps the existing path. The group join uses the
//! trusted statement group's codes and the registered G_u shape. Column rows
//! come either from the reads verified against the trusted reader root or from
//! an explicit, unauthenticated fixture; both stay pending, and only the
//! verified source can satisfy the FOLD gate. The statement and
//! runtime layout origins are plan-provided test inputs in this isolated
//! layer (not production-authenticated sources), the
//! proof-byte answer decoder is not connected, and no production
//! statement-authentication, relation or FOLD closure is claimed. With a G_u
//! shape `fold_bind_row` fixes one explicitly unauthenticated input-row fixture
//! to the session's group, layout, shape and existing membership before the
//! epsilon challenge; `start_fold` consumes that fixed row; `fold_rounds_start`
//! begins the rounds from it; `receive_fold_round` absorbs the complete
//! two-value message before drawing the same round's challenge; and
//! `fold_finish` stores the folded coefficient, mask/fresh, kept, target and
//! identity constraint. Proof-byte encoding and a real production chain source
//! are not connected, and the next IOPP/CS stage fails closed. The protocol
//! plan's per-round error-parameter binding, mask query/authentication, real
//! statement/group source and canonical proof bytes are also not connected. The
//! next wiring order must first bind the trusted statement/group and the plan's
//! per-round error parameters, then add canonical message bytes and the mask
//! query/opening checks, and only then connect the next stage. Until then this
//! session does not accept a proof.

use std::sync::{Arc, Mutex};

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::pcs::request::binding::{BindingParameters, BindingStatement};
use crate::pcs::request::chain::{
    Chain, KIND_ALGEBRAIC, KIND_EPSILON, KIND_EMPTY, KIND_POSITIONS, KIND_SEAL, KIND_SEED,
};
use crate::pcs::request::field::{Fp, Fp3};
use crate::pcs::request::pow::{search_nonce_for_plan, verify_nonce, MAX_WORK_BITS};
use crate::pcs::request::gu::{GuMembership, GuNotice, GuOracleShape, GuRegistry, GuShape};
use crate::pcs::request::relation::{
    column_plan_columns, column_row, column_target, foldeq_weights, gu_layout_fault,
    oracle_sample_row, record_row, sample_cube_of,
    sample_row, GuRelationRow,
};
use crate::pcs::request::runtime_tree::{RuntimeTreeOpening, RuntimeTreePlan};
use crate::pcs::request::fold::{
    fold_mask_shape_fault, fold_output, fold_start_verify, fold_start_verify_with_groups,
    FoldGroupLayout, FoldInputRow, FoldOutput, FoldRoundVerifier, FoldShape, FoldStartMessage,
    MaskAuthentication, RuntimeOracles,
    VerifiedStatement,
};
use crate::pcs::request::read::{
    draw_positions, unique_sorted, HidingCharge, HidingState, ReadLedger, VerifierReads,
};
use crate::pcs::request::table::{
    index_name, record_seeds, ColumnOpening, IndexName, IndexReader, SampleSeed,
};
use crate::primitives::codec::CanonicalWriter;
use crate::primitives::transcript::chain_start;

/// The binder that both sides build from the complete first query. A session
/// accepts only a chain whose label and start value are the binder's; a
/// pre-built chain from another prefix is rejected.
#[derive(Clone, Debug)]
pub struct FoldeqBinder {
    label: u64,
    parameters: Digest,
    first_query: Vec<u8>,
    start: Digest,
    random_bytes: Option<Digest>,
}

impl FoldeqBinder {
    pub fn new(label: u64, parameters: Digest, plan: &FoldeqPlan) -> Result<Self> {
        Self::new_with_random_bytes(label, parameters, plan, None)
    }

    /// Builds the binder with the chain's random bytes. The salted chain's
    /// start and its run's value follow the same bytes; the trees that the
    /// chain sends use that run as their start.
    pub fn new_with_random_bytes(
        label: u64,
        parameters: Digest,
        plan: &FoldeqPlan,
        random_bytes: Option<Digest>,
    ) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if parameters != plan.reader.configuration {
            return Err(Error::StatementMismatch);
        }
        let first_query = plan.first_query();
        let start = chain_start(
            label,
            &parameters,
            &first_query,
            &[],
            random_bytes.as_ref(),
        )?;
        Ok(Self {
            label,
            parameters,
            first_query,
            start,
            random_bytes,
        })
    }

    /// The chain's run's value when its first query carries random bytes.
    pub fn run(&self) -> Option<Digest> {
        self.random_bytes
            .as_ref()
            .and_then(|bytes| {
                crate::primitives::transcript::run_value(
                    self.label,
                    &self.parameters,
                    bytes,
                )
                .ok()
            })
    }

    pub fn random_bytes(&self) -> Option<&Digest> {
        self.random_bytes.as_ref()
    }

    pub fn first_query(&self) -> &[u8] {
        &self.first_query
    }

    pub fn start(&self) -> &Digest {
        &self.start
    }

    pub fn chain(&self) -> Result<Chain> {
        Chain::begin(
            self.label,
            &self.parameters,
            &self.first_query,
            &[],
            self.random_bytes.as_ref(),
        )
    }
}

/// The provider's G_u message: the root the provider sends after its durable
/// hiding pre-charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuMessage {
    pub root: Digest,
}

/// The provider's side of the G_u/column phase. `begin` must charge the
/// provider hiding state before `send_gu` returns a message; an uncharged
/// provider cannot send. The charge is in-memory here; durable cross-process
/// storage and rollback protection remain open.
pub struct ProverFoldeq {
    hiding: HidingState,
    table: Digest,
    lifecycle: u64,
    columns: usize,
    charged: bool,
}

impl ProverFoldeq {
    /// Bind the provider state to the actual table root, the lifecycle and the
    /// plan's column count. The maximum is still a caller-provided value; its
    /// derivation from the encoded tail/opening budgets is open.
    pub fn new(
        hiding: HidingState,
        reader: &IndexReader,
        lifecycle: u64,
        columns: usize,
    ) -> Result<Self> {
        if hiding.table() != reader.root
            || hiding.lifecycle() != lifecycle
            || columns == 0
        {
            return Err(Error::StatementMismatch);
        }
        Ok(Self {
            hiding,
            table: reader.root,
            lifecycle,
            columns,
            charged: false,
        })
    }

    /// Charge the provider hiding state before G_u and any column exposure.
    /// This is an in-memory update, not a durable pre-charge; the durable
    /// provider ledger (IndexHolder-like) is still open. A second call fails;
    /// an over-budget charge fails before any send. The returned token is
    /// required by the charged column-open path.
    pub fn begin(&mut self) -> Result<HidingCharge> {
        if self.charged {
            return Err(Error::InvalidSessionState);
        }
        let charge = self
            .hiding
            .begin_charge(self.table, self.lifecycle, self.columns)?;
        self.charged = true;
        Ok(charge)
    }

    /// Return the G_u message only after the hiding pre-charge.
    pub fn send_gu(&self, root: &Digest) -> Result<GuMessage> {
        if !self.charged || *root == Digest::ZERO {
            return Err(Error::InvalidSessionState);
        }
        Ok(GuMessage { root: *root })
    }

    pub fn used(&self) -> u64 {
        self.hiding.used()
    }
}

/// A LIN-style gamma: the suffix of a caller-drawn psi, starting at `k_lo`.
/// The FOLDEQ session does not assume this caller; a direct gamma is also
/// accepted. The caller owns the psi draw and its round.
fn lin_gamma(psi: &[Fp3], k_lo: u32, columns: u32) -> Result<Vec<Fp3>> {
    let start = k_lo as usize;
    let end = start
        .checked_add(columns as usize)
        .ok_or(Error::ResourceLimit)?;
    if end > psi.len() {
        return Err(Error::ShapeMismatch);
    }
    Ok(psi[start..end].to_vec())
}

/// The trusted challenge path of one FOLDEQ plan. A LIN caller fixes the
/// LinShape's k_lo/k_col; a caller with another reference path fixes the
/// direct length. The plan chooses the path, never the proof or an untrusted
/// caller, and the first query binds these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GammaPath {
    Lin { shape: LinShapeBinding },
    Direct { columns: u32 },
}

/// The verifier's binding of the caller's LinShape for the gamma path: the
/// matrix rows and the low/column counts. The verifier constructs it from its
/// own trusted LinShape and the first query binds these fields. Carrying the
/// fields (not a caller-supplied digest) keeps the values checkable; the
/// trusted source itself remains a verifier/plan prerequisite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinShapeBinding {
    pub matrix_rows: u64,
    pub k_lo: u32,
    pub k_col: u32,
}

/// The maximum canonical statement length admitted into a first query.
pub const MAX_BINDING_STATEMENT_BYTES: usize = 1 << 20;

/// The verifier-side typed binding prefix for one FOLDEQ plan.
///
/// The parameters digest is the table/chain configuration; the statement is
/// canonically encoded from the typed 2b statement after the reader exists.
/// The statement's table identity must equal the reader's actual root, shape,
/// configuration and record. The bytes are constructed here, never supplied
/// as an untyped prover string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldeqBinding {
    parameters_digest: Digest,
    version_word: u32,
    protocol_identity: Digest,
    statement_bytes: Vec<u8>,
    table_identity: Digest,
}

impl FoldeqBinding {
    pub fn new(
        parameters: &BindingParameters,
        statement: BindingStatement,
        reader: &IndexReader,
    ) -> Result<Self> {
        parameters.validate()?;
        statement.validate()?;
        if statement.version_word != parameters.version_word
            || statement.protocol_identity != parameters.protocol_identity
        {
            return Err(Error::StatementMismatch);
        }
        let table_identity = reader_table_identity(reader)?;
        if statement.table_identity != table_identity {
            return Err(Error::StatementMismatch);
        }
        let statement_bytes = statement.canonical_bytes()?;
        if statement_bytes.is_empty() || statement_bytes.len() > MAX_BINDING_STATEMENT_BYTES {
            return Err(Error::ResourceLimit);
        }
        Ok(Self {
            parameters_digest: parameters.digest()?,
            version_word: parameters.version_word,
            protocol_identity: parameters.protocol_identity,
            statement_bytes,
            table_identity,
        })
    }

    pub fn parameters_digest(&self) -> Digest {
        self.parameters_digest
    }

    pub fn version_word(&self) -> u32 {
        self.version_word
    }

    pub fn protocol_identity(&self) -> Digest {
        self.protocol_identity
    }

    pub fn statement_bytes(&self) -> &[u8] {
        &self.statement_bytes
    }

    pub fn table_identity(&self) -> Digest {
        self.table_identity
    }
}

/// The reader's actual table identity: root, shape, configuration and record.
/// This is the value a binding statement must name; it is not a separate
/// caller-supplied identity.
pub fn reader_table_identity(reader: &IndexReader) -> Result<Digest> {
    if reader.fault().is_some() {
        return Err(Error::InvalidConfig);
    }
    let mut writer = CanonicalWriter::new();
    writer.put_digest(&reader.root);
    writer.put_digest(&reader.configuration);
    for value in [
        reader.shape.code.log_n as u64,
        reader.shape.code.ell,
        reader.shape.code.r,
        reader.shape.v as u64,
        reader.shape.answers as u64,
        reader.shape.salt_bytes as u64,
    ] {
        writer.put_u64(value);
    }
    writer.put_bytes(
        &IndexName {
            commitment: reader.commitment(),
            record: reader.record.clone(),
        }
        .encode(),
    );
    Ok(digest_bytes(&writer.into_bytes()))
}

/// Direction of a graph edge. The candidate step plan currently only carries
/// forward edges; the tag is explicit so a future direction cannot be implied
/// by field order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeDirection {
    Forward,
}

/// A typed layout or slice on one edge. It mirrors the selected static plan's
/// layout vocabulary without importing the plan module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdgeLayout {
    Flat,
    Range { start: u64, count: u64 },
    Rows { row_width: u64, start: u64, count: u64 },
    Columns { row_width: u64, start: u64, count: u64 },
    Transpose { rows: u64, cols: u64 },
    Broadcast { rows: u64 },
    Chain(Vec<EdgeLayout>),
}

impl EdgeLayout {
    fn check(&self, depth: u32) -> Result<()> {
        if depth > 16 {
            return Err(Error::ResourceLimit);
        }
        match self {
            Self::Flat => Ok(()),
            Self::Range { count, .. } => {
                if *count == 0 {
                    return Err(Error::InvalidConfig);
                }
                Ok(())
            }
            Self::Rows {
                row_width, count, ..
            }
            | Self::Columns {
                row_width, count, ..
            } => {
                if *row_width == 0 || *count == 0 {
                    return Err(Error::InvalidConfig);
                }
                Ok(())
            }
            Self::Transpose { rows, cols } => {
                if *rows == 0 || *cols == 0 {
                    return Err(Error::InvalidConfig);
                }
                Ok(())
            }
            Self::Broadcast { rows } => {
                if *rows == 0 {
                    return Err(Error::InvalidConfig);
                }
                Ok(())
            }
            Self::Chain(parts) => {
                if parts.is_empty() || parts.len() > 16 {
                    return Err(Error::InvalidConfig);
                }
                for part in parts {
                    part.check(depth + 1)?;
                }
                Ok(())
            }
        }
    }

    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Flat => writer.put_u64(0),
            Self::Range { start, count } => {
                writer.put_u64(1);
                writer.put_u64(*start);
                writer.put_u64(*count);
            }
            Self::Rows {
                row_width,
                start,
                count,
            } => {
                writer.put_u64(2);
                writer.put_u64(*row_width);
                writer.put_u64(*start);
                writer.put_u64(*count);
            }
            Self::Columns {
                row_width,
                start,
                count,
            } => {
                writer.put_u64(3);
                writer.put_u64(*row_width);
                writer.put_u64(*start);
                writer.put_u64(*count);
            }
            Self::Transpose { rows, cols } => {
                writer.put_u64(4);
                writer.put_u64(*rows);
                writer.put_u64(*cols);
            }
            Self::Broadcast { rows } => {
                writer.put_u64(5);
                writer.put_u64(*rows);
            }
            Self::Chain(parts) => {
                writer.put_u64(6);
                writer.put_u64(parts.len() as u64);
                for part in parts {
                    part.encode(writer);
                }
            }
        }
    }
}

/// A previous-step output edge. It has no asset identity: the source is a
/// producer output, not an asset reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrevEdge {
    pub producer_step: u64,
    pub producer_output: u64,
    pub consumer_step: u64,
    pub input_port: u64,
    pub direction: EdgeDirection,
    pub layout: EdgeLayout,
}

/// An asset-reference edge. It has no producer step/output: the source is an
/// asset identity, not a previous-step output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetEdge {
    pub consumer_step: u64,
    pub input_port: u64,
    pub direction: EdgeDirection,
    pub asset_id: u64,
    pub path: Vec<u8>,
    pub start: Digest,
    pub root: Digest,
    pub rows: u64,
    pub row_width: u64,
    pub element_count: u64,
    pub layout: EdgeLayout,
}

/// A typed edge declaration. `NotAnEdge` is explicit for sessions that are
/// not graph-edge sessions; the two bound variants keep previous-step and
/// asset-reference semantics separate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoldeqEdgeBinding {
    NotAnEdge,
    Prev(PrevEdge),
    Asset(AssetEdge),
}

impl FoldeqEdgeBinding {
    pub fn is_bound(&self) -> bool {
        !matches!(self, Self::NotAnEdge)
    }

    pub fn check(&self) -> Result<()> {
        match self {
            Self::NotAnEdge => Ok(()),
            Self::Prev(edge) => {
                if edge.producer_step == edge.consumer_step {
                    return Err(Error::InvalidConfig);
                }
                edge.layout.check(0)
            }
            Self::Asset(edge) => {
                if edge.path.is_empty()
                    || edge.rows == 0
                    || edge.row_width == 0
                    || edge.element_count == 0
                    || edge.start == Digest::ZERO
                    || edge.root == Digest::ZERO
                {
                    return Err(Error::InvalidConfig);
                }
                edge.layout.check(0)
            }
        }
    }

    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::NotAnEdge => {}
            Self::Prev(edge) => {
                writer.put_u64(1);
                writer.put_u64(edge.producer_step);
                writer.put_u64(edge.producer_output);
                writer.put_u64(edge.consumer_step);
                writer.put_u64(edge.input_port);
                writer.put_u64(0);
                edge.layout.encode(writer);
            }
            Self::Asset(edge) => {
                writer.put_u64(2);
                writer.put_u64(edge.consumer_step);
                writer.put_u64(edge.input_port);
                writer.put_u64(0);
                writer.put_u64(edge.asset_id);
                writer.put_bytes(&edge.path);
                writer.put_digest(&edge.start);
                writer.put_digest(&edge.root);
                writer.put_u64(edge.rows);
                writer.put_u64(edge.row_width);
                writer.put_u64(edge.element_count);
                edge.layout.encode(writer);
            }
        }
    }
}

/// The trusted plan of the shared chain's Omega oracle message. Omega is a
/// static member of the FOLD group; its vector is read by the FOLD row and
/// its message carries one sample answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OmegaShapePlan {
    pub oracle: u64,
    pub group: u64,
    pub tree: RuntimeTreePlan,
    pub u: GuOracleShape,
    pub rows_high: bool,
    pub matrix_rows: u64,
    pub k_lo: u32,
}

impl OmegaShapePlan {
    pub fn fault(&self) -> Option<&'static str> {
        if self.u.fault().is_some() {
            return Some("the Omega oracle shape is not an oracle shape");
        }
        if self.u.mask_rows != 0 || self.u.answers != 1 {
            return Some("the Omega oracle is not a maskless single-answer message");
        }
        if self.tree.fault().is_some() {
            return Some("the Omega tree is not a runtime tree");
        }
        if self.tree.parts.len() != 1 {
            return Some("the Omega tree is not one part");
        }
        let part = &self.tree.parts[0];
        let main_symbols = match self.u.main_rows.checked_mul(self.u.main_deg as u64) {
            Some(value) => value,
            None => return Some("the Omega main width overflows"),
        };
        if part.main_symbols != main_symbols || part.mask_symbols != 0 {
            return Some("the Omega part's widths are not its oracle shape's");
        }
        if self.tree.log_n != self.u.code.log_n {
            return Some("the Omega tree depth is not the oracle code's depth");
        }
        if self.matrix_rows < 1
            || !self.matrix_rows.is_power_of_two()
            || self.matrix_rows > (1u64 << 30)
            || self.k_lo >= 32
        {
            return Some("the Omega vector shape is not executable");
        }
        if self.matrix_rows + self.u.answers as u64 > self.u.code.ell {
            return Some("the Omega vector does not fit its message beside the pad");
        }
        None
    }

    fn encode(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        writer.put_u64(self.oracle);
        writer.put_u64(self.group);
        let tree = self.tree.encode();
        writer.put_u64(tree.len() as u64);
        writer.put_bytes(&tree);
        writer.put_u64(if self.rows_high { 1 } else { 0 });
        writer.put_u64(self.u.code.log_n as u64);
        writer.put_u64(self.u.code.ell);
        writer.put_u64(self.u.code.r);
        writer.put_u64(self.u.main_rows);
        writer.put_u64(self.u.main_deg as u64);
        writer.put_u64(self.u.mask_rows);
        writer.put_u64(self.u.answers as u64);
        writer.put_u64(self.matrix_rows);
        writer.put_u64(self.k_lo as u64);
        writer.into_bytes()
    }
}

/// The verifier's FOLDEQ plan. The reader is the committed table, the fold and
/// runtime mask-tree layouts and the group registry come from the statement,
/// and the seed/read budgets and column draw count come from the plan. None of
/// these is taken from a proof. The sampling distances are the first-query
/// exception: they stay plan values and are not in the first-query bytes.
#[derive(Clone, Debug)]
pub struct FoldeqPlan {
    pub reader: IndexReader,
    /// The required typed binding for this plan. The statement bytes are part
    /// of the first query; the parameters digest must equal the reader's
    /// configuration. There is no unbound production form.
    pub binding: FoldeqBinding,
    /// The typed graph edge, if this is an edge session. `NotAnEdge` is
    /// explicit and is the only non-edge form.
    pub edge: FoldeqEdgeBinding,
    pub fold: FoldShape,
    pub statement: Arc<VerifiedStatement>,
    pub oracles: Arc<Mutex<RuntimeOracles>>,
    pub mask_tree_number: u64,
    pub group_id: u64,
    pub leaf_class: u32,
    pub seed_budget: usize,
    pub ledger: Arc<Mutex<ReadLedger>>,
    pub column_draws: usize,
    pub mask_query_count: usize,
    pub gamma_path: GammaPath,
    /// The trusted G_u shape when this plan carries the isolated G_u relation
    /// session. It is a plan/statement value, not a proof report.
    pub gu: Option<GuShape>,
    /// The trusted fixed group layout for the isolated FOLD input row. It is a
    /// fixture supplied by the plan, not a proof field.
    pub fold_layout: Option<FoldGroupLayout>,
    /// The shared chain's Omega oracle plan, when the chain carries it.
    pub omega: Option<OmegaShapePlan>,
}

impl FoldeqPlan {
    pub fn fault(&self) -> Option<&'static str> {
        if let Some(fault) = self.reader.fault() {
            return Some(fault);
        }
        if self.binding.parameters_digest != self.reader.configuration {
            return Some("the binding parameters are not the table configuration");
        }
        let expected_identity = match reader_table_identity(&self.reader) {
            Ok(identity) => identity,
            Err(_) => return Some("the reader has no table identity"),
        };
        if self.binding.table_identity != expected_identity {
            return Some("the binding table identity is not the reader's");
        }
        if self.binding.statement_bytes.is_empty()
            || self.binding.statement_bytes.len() > MAX_BINDING_STATEMENT_BYTES
        {
            return Some("the binding statement has no admitted bytes");
        }
        if self.edge.check().is_err() {
            return Some("the edge binding is not well formed");
        }
        if let Some(fault) = fold_mask_shape_fault(&self.fold) {
            return Some(fault);
        }
        let tree = match self.statement.mask_tree(self.mask_tree_number) {
            Some(tree) => tree,
            None => return Some("the plan names a runtime tree the statement does not hold"),
        };
        if let Some(fault) = tree.fault() {
            return Some(fault);
        }
        let rows = match self.fold.mask_rows() {
            Ok(rows) => rows,
            Err(_) => return Some("the fold shape has no mask rows"),
        };
        if tree.mask_rows != rows
            || tree.log_n != self.fold.mask_code.log_n
            || tree.leaf_class != self.fold.mask_leaf_class
        {
            return Some("the runtime mask tree does not match the fold shape");
        }
        let group = match self.statement.groups().get(self.group_id) {
            Some(group) => group,
            None => return Some("the plan names a group the verifier does not hold"),
        };
        if group.members.is_empty() {
            return Some("the plan's group has no members");
        }
        if self.column_draws == 0 {
            return Some("no column is drawn");
        }
        let draws = match u64::try_from(self.column_draws) {
            Ok(draws) => draws,
            Err(_) => return Some("the column draw count does not fit"),
        };
        let max_reads = match self.ledger.lock() {
            Ok(ledger) => ledger.max_reads(),
            Err(_) => return Some("the read ledger is poisoned"),
        };
        if draws > max_reads {
            return Some("the column draws exceed the read budget");
        }
        let answers = self.reader.shape.answers as usize;
        if self.seed_budget < answers {
            return Some("the seed budget is below the answer count");
        }
        if let Some(gu) = &self.gu {
            if gu.fault().is_some() {
                return Some("the G_u shape is not valid");
            }
            if gu.index != self.reader.shape {
                return Some("the G_u index shape is not the plan's reader shape");
            }
            if gu.group != self.group_id {
                return Some("the G_u group is not the FOLD group");
            }
            if self.statement.groups().get(gu.group).is_none() {
                return Some("the G_u group is not in the statement");
            }
            if crate::pcs::request::relation::gu_layout_fault(gu).is_some() {
                return Some("the G_u layout is not executable");
            }
            if let Some(layout) = &self.fold_layout {
                if layout.group != self.group_id {
                    return Some("the FOLD layout group is not the plan group");
                }
                if layout.fault(&self.fold).is_some() {
                    return Some("the FOLD layout does not match the fold shape");
                }
            }
        }
        if let Some(omega) = &self.omega {
            if omega.fault().is_some() {
                return Some("the Omega shape is not valid");
            }
            if self.gu.is_none() {
                return Some("the Omega message has no G_u message");
            }
            if omega.group != self.group_id {
                return Some("the Omega group is not the FOLD group");
            }
            if omega.oracle == self.gu.as_ref().expect("checked").oracle {
                return Some("the Omega oracle is the G_u oracle");
            }
            let group = match self.statement.groups().get(self.group_id) {
                Some(group) => group,
                None => return Some("the plan names a group the verifier does not hold"),
            };
            if !group.members.contains(&omega.oracle) || group.fresh == Some(omega.oracle) {
                return Some("the Omega oracle is not a static member of the FOLD group");
            }
        }
        match self.gamma_path {
            GammaPath::Lin { shape } => {
                if shape.matrix_rows < 1
                    || shape.matrix_rows > (1u64 << 30)
                    || shape.k_lo > 30
                    || shape.k_col > 60
                {
                    return Some("the LIN shape binding has invalid numbers");
                }
                let expected = match shape.k_lo.checked_add(self.reader.shape.v) {
                    Some(expected) => expected,
                    None => return Some("the LIN challenge path overflows"),
                };
                if shape.k_col != expected {
                    return Some("the LIN challenge path does not end at the table width");
                }
                // A shift that loses high bits is not a fit: checked_shl only
                // rejects an out-of-range shift amount, so use a checked
                // multiply by the explicit power of two instead.
                let factor = match 1u64.checked_shl(shape.k_lo) {
                    Some(factor) => factor,
                    None => return Some("the matrix width overflows"),
                };
                let width = match shape.matrix_rows.checked_mul(factor) {
                    Some(width) => width,
                    None => return Some("the matrix width overflows"),
                };
                if width > self.reader.shape.plain() {
                    return Some("the matrix does not fit the table's plain columns");
                }
            }
            GammaPath::Direct { columns } => {
                if columns != self.reader.shape.v {
                    return Some("the direct challenge length is not the table width");
                }
            }
        }
        None
    }

    /// The bytes the first query binds: the complete table name (commitment and
    /// record), before any challenge that reads it. Distances are not part of
    /// the first query; they stay plan values.
    pub fn first_query(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::new();
        // The complete table name: commitment and record.
        writer.put_bytes(
            &IndexName {
                commitment: self.reader.commitment(),
                record: self.reader.record.clone(),
            }
            .encode(),
        );
        // The typed binding statement, after the reader root/record and before
        // the verified statement id. The parameters digest is the binder's
        // parameter value, not these bytes.
        writer.put_bytes(&self.binding.statement_bytes);
        // The typed edge identity, after the statement and before all shapes
        // and challenge fields. A non-edge session writes no edge marker, so
        // the 2f single-step first query is unchanged.
        self.edge.encode(&mut writer);
        // The trusted table and fold shapes. Distances are the first-query
        // exception and stay plan values; the configuration is the binder's
        // parameters digest, not this statement.
        for value in [
            self.reader.shape.code.log_n as u64,
            self.reader.shape.code.ell,
            self.reader.shape.code.r,
            self.reader.shape.v as u64,
            self.reader.shape.answers as u64,
            self.reader.shape.salt_bytes as u64,
            self.fold.code.log_n as u64,
            self.fold.code.ell,
            self.fold.code.r,
            self.fold.mask_code.log_n as u64,
            self.fold.mask_code.ell,
            self.fold.mask_code.r,
            self.fold.k as u64,
            self.fold.h as u64,
            self.fold.mask_leaf_class as u64,
        ] {
            writer.put_u64(value);
        }
        // The verified statement (groups, runtime trees and their starts), the
        // expected tree/group/leaf numbers and the complete query plan.
        writer.put_digest(self.statement.id());
        for value in [
            self.mask_tree_number,
            self.group_id,
            self.leaf_class as u64,
            self.seed_budget as u64,
            self.column_draws as u64,
        ] {
            writer.put_u64(value);
        }
        let max_reads = self
            .ledger
            .lock()
            .map(|ledger| ledger.max_reads())
            .unwrap_or(0);
        writer.put_u64(max_reads);
        writer.put_u64(self.mask_query_count as u64);
        match self.gamma_path {
            GammaPath::Lin { shape } => {
                writer.put_u64(1);
                writer.put_u64(shape.matrix_rows);
                writer.put_u64(shape.k_lo as u64);
                writer.put_u64(shape.k_col as u64);
            }
            GammaPath::Direct { columns } => {
                writer.put_u64(2);
                writer.put_u64(columns as u64);
            }
        }
        match &self.gu {
            Some(gu) => {
                writer.put_u64(1);
                let bytes = gu.encode();
                writer.put_u64(bytes.len() as u64);
                writer.put_bytes(&bytes);
            }
            None => writer.put_u64(0),
        }
        match &self.omega {
            Some(omega) => {
                writer.put_u64(1);
                let bytes = omega.encode();
                writer.put_u64(bytes.len() as u64);
                writer.put_bytes(&bytes);
            }
            None => writer.put_u64(0),
        }
        writer.into_bytes()
    }

    pub fn check_first_query(&self, sent: &[u8]) -> Result<()> {
        if sent != self.first_query() {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        record_seeds(&self.reader.shape, &self.reader.name(), self.seed_budget)
    }

    pub fn draw_queries(&self, chain: &mut Chain) -> Result<Vec<u64>> {
        draw_positions(chain, self.column_draws, self.reader.shape.code.log_n)
    }
}

/// Where the joined G_u column rows came from. A fixture is useful for
/// isolated evaluation but is not a reader authentication result and cannot
/// satisfy the FOLD gate; only the verified read path can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuColumnSource {
    Fixture,
    Verified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    FirstQuery,
    Seeds,
    Gamma,
    Gu,
    GuAnswers,
    GuRecord,
    GuJoined,
    Queried,
    Read,
    FoldStarted,
    FoldRounds,
    FoldFinished,
    Omega,
    OmegaAnswers,
    OmegaJoined,
    ZetaDrawn,
    BatDrawn,
    MaskQueried,
    MaskOpened,
}

/// The verifier's FOLDEQ session. It owns the chain, the plan, the mask spend
/// ledger and the read store; the prover's public messages and openings are
/// passed in. This is a joint-development harness, not a production verifier.
pub struct FoldeqSession {
    plan: FoldeqPlan,
    chain: Chain,
    reads: VerifierReads,
    stage: Stage,
    gamma: Option<Vec<Fp3>>,
    psi: Option<Vec<Fp3>>,
    gu_root: Option<Digest>,
    start: Option<FoldStartMessage>,
    epsilon: Option<Fp3>,
    queries: Vec<u64>,
    mask_queries: Vec<u64>,
    gu_registry: GuRegistry,
    gu_seeds: Vec<SampleSeed>,
    gu_answers: Vec<Fp3>,
    gu_answered: usize,
    gu_membership: Option<GuMembership>,
    gu_read_columns: Option<Vec<Vec<Fp>>>,
    gu_fixture_columns: Option<Vec<(u64, Vec<Fp3>)>>,
    gu_columns_source: Option<GuColumnSource>,
    gu_column_targets: Vec<Fp3>,
    gu_rows: Vec<GuRelationRow>,
    omega_root: Option<Digest>,
    omega_seeds: Vec<SampleSeed>,
    omega_answers: Vec<Fp3>,
    omega_answered: usize,
    omega_rows: Vec<GuRelationRow>,
    omega_joined: bool,
    zeta: Option<Vec<Fp3>>,
    bat_x: Option<Fp3>,
    fold_row: Option<FoldInputRow>,
    fold_claim: Option<FoldRoundVerifier>,
    fold_gamma: Vec<Fp3>,
    fold_rounds: u32,
    fold_result: Option<FoldOutput>,
}

impl FoldeqSession {
    /// Open a session only from a chain built by the binder. A chain from a
    /// different first-query prefix, or one that has already advanced, is
    /// rejected.
    pub fn open(plan: FoldeqPlan, binder: FoldeqBinder, chain: Chain) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if binder.first_query() != plan.first_query().as_slice()
            || chain.label() != binder.label
            || chain.value() != binder.start()
            || chain.rounds() != 0
        {
            return Err(Error::StatementMismatch);
        }
        let reads = VerifierReads::new(Arc::clone(&plan.ledger));
        Ok(Self {
            plan,
            chain,
            reads,
            stage: Stage::FirstQuery,
            gamma: None,
            psi: None,
            gu_root: None,
            start: None,
            epsilon: None,
            queries: Vec::new(),
            mask_queries: Vec::new(),
            gu_registry: GuRegistry::new(),
            gu_seeds: Vec::new(),
            gu_answers: Vec::new(),
            gu_answered: 0,
            gu_membership: None,
            gu_read_columns: None,
            gu_fixture_columns: None,
            gu_columns_source: None,
            gu_column_targets: Vec::new(),
            gu_rows: Vec::new(),
            omega_root: None,
            omega_seeds: Vec::new(),
            omega_answers: Vec::new(),
            omega_answered: 0,
            omega_rows: Vec::new(),
            omega_joined: false,
            zeta: None,
            bat_x: None,
            fold_row: None,
            fold_claim: None,
            fold_gamma: Vec::new(),
            fold_rounds: 0,
            fold_result: None,
        })
    }

    /// Opens the isolated FOLD-only fixture session on a chain already advanced
    /// by a fixture prior stage. This entry bypasses the first-query binder and
    /// the G_u/record/column prefix: it exists for one isolated component
    /// handoff, cannot accept a proof, and is not a production entry. The
    /// ordinary `open` remains the only full-session constructor. The caller
    /// must pass the chain returned by the prior stage; this constructor cannot
    /// verify that origin. The plan's reader and binding are a valid but inert
    /// scaffold here: the FOLD start/rounds read only the fold shape, group,
    /// statement, runtime registry and layout. A plan with a G_u shape is
    /// rejected because its start path needs joined membership and verified
    /// read state that this entry does not produce.
    pub fn open_fold_fixture(plan: FoldeqPlan, chain: Chain) -> Result<Self> {
        if plan.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        if plan.gu.is_some() {
            return Err(Error::UnsupportedPrimitive);
        }
        let layout = plan.fold_layout.as_ref().ok_or(Error::InvalidConfig)?;
        if layout.fault(&plan.fold).is_some() || layout.group != plan.group_id {
            return Err(Error::InvalidConfig);
        }
        if chain.rounds() == 0 {
            return Err(Error::InvalidSessionState);
        }
        let reads = VerifierReads::new(Arc::clone(&plan.ledger));
        Ok(Self {
            plan,
            chain,
            reads,
            stage: Stage::Read,
            gamma: None,
            psi: None,
            gu_root: None,
            start: None,
            epsilon: None,
            queries: Vec::new(),
            mask_queries: Vec::new(),
            gu_registry: GuRegistry::new(),
            gu_seeds: Vec::new(),
            gu_answers: Vec::new(),
            gu_answered: 0,
            gu_membership: None,
            gu_read_columns: None,
            gu_fixture_columns: None,
            gu_columns_source: None,
            gu_column_targets: Vec::new(),
            gu_rows: Vec::new(),
            omega_root: None,
            omega_seeds: Vec::new(),
            omega_answers: Vec::new(),
            omega_answered: 0,
            omega_rows: Vec::new(),
            omega_joined: false,
            zeta: None,
            bat_x: None,
            fold_row: None,
            fold_claim: None,
            fold_gamma: Vec::new(),
            fold_rounds: 0,
            fold_result: None,
        })
    }

    pub fn plan(&self) -> &FoldeqPlan {
        &self.plan
    }

    pub fn queries(&self) -> &[u64] {
        &self.queries
    }

    pub fn mask_queries(&self) -> &[u64] {
        &self.mask_queries
    }

    pub fn gamma(&self) -> Option<&[Fp3]> {
        self.gamma.as_deref()
    }

    /// The complete LIN point drawn before the gamma suffix. It is present
    /// only on the LIN challenge path.
    pub fn psi(&self) -> Option<&[Fp3]> {
        self.psi.as_deref()
    }

    /// The seed set drawn from the received G_u root's SEND round. It is
    /// session state, not a proof field.
    pub fn gu_shared_seeds(&self) -> &[SampleSeed] {
        &self.gu_seeds
    }

    pub fn gu_root(&self) -> Option<&Digest> {
        self.gu_root.as_ref()
    }

    /// The current chain value. This is session state for inspection, not a
    /// proof field and not a public transcript artifact.
    /// The session chain at the current stage. This is an isolated inspection
    /// accessor for the candidate G_u declaration combination; it does not
    /// expose a production acceptance path.
    pub fn chain(&self) -> &crate::pcs::request::chain::Chain {
        &self.chain
    }

    /// Transfers chain ownership to the BAT stage after the G_u/record/column
    /// declarations are joined. The session is consumed, so the old handle
    /// cannot advance the chain independently. A missing join or an earlier
    /// stage is rejected. The returned chain is the same state the session
    /// ended with; this is an ownership transfer, not a copied digest.
    pub fn take_chain_for_bat(self) -> Result<crate::pcs::request::chain::Chain> {
        if self.stage != Stage::Read
            || self.gu_membership.is_none()
            || self.gu_columns_source.is_none()
        {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.chain)
    }

    pub fn transcript_value(&self) -> &Digest {
        self.chain.value()
    }

    pub fn epsilon(&self) -> Option<Fp3> {
        self.epsilon
    }

    pub fn start(&self) -> Option<&FoldStartMessage> {
        self.start.as_ref()
    }

    pub fn bind_first_query(&mut self, sent: &[u8]) -> Result<()> {
        if self.stage != Stage::FirstQuery {
            return Err(Error::InvalidSessionState);
        }
        self.plan.check_first_query(sent)?;
        self.stage = Stage::Seeds;
        Ok(())
    }

    pub fn seeds(&self) -> Result<Vec<SampleSeed>> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        self.plan.seeds()
    }

    /// Draw the challenge along the plan's fixed path. The path and its
    /// numbers are trusted-plan values bound by the first query; the caller
    /// cannot choose LIN versus direct or replace the offset, and the session
    /// owns the draw so no caller value or forged psi source is accepted.
    pub fn draw_gamma(&mut self) -> Result<()> {
        if self.stage != Stage::Seeds {
            return Err(Error::InvalidSessionState);
        }
        let gamma = match self.plan.gamma_path {
            GammaPath::Lin { shape } => {
                let psi = self.draw_challenge(shape.k_col as usize)?;
                let gamma = lin_gamma(&psi, shape.k_lo, self.plan.reader.shape.v)?;
                self.psi = Some(psi);
                gamma
            }
            GammaPath::Direct { columns } => self.draw_challenge(columns as usize)?,
        };
        self.set_gamma(gamma)
    }

    fn draw_challenge(&mut self, count: usize) -> Result<Vec<Fp3>> {
        let prefix = self.chain.message().prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..count {
            values.push(self.chain.draw_e()?);
        }
        Ok(values)
    }

    fn set_gamma(&mut self, gamma: Vec<Fp3>) -> Result<()> {
        self.gamma = Some(gamma);
        self.stage = Stage::Gamma;
        Ok(())
    }

    /// Bind the prover's G_u SEND root. The root is a public message value;
    /// the prover's folded table and its OOD/record relations are verified by
    /// the next stages and remain open.
    pub fn send_gu(&mut self, root: &Digest) -> Result<()> {
        if self.stage != Stage::Gamma {
            return Err(Error::InvalidSessionState);
        }
        if *root == Digest::ZERO {
            return Err(Error::InvalidProof);
        }
        let mut message = self.chain.message();
        for chunk in root.0.chunks(8) {
            let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
            message.push_u64(word);
        }
        // With a G_u shape the root opens the seed round: the verifier's
        // message is the seed set, drawn by `shared_seeds`. Without one the
        // root keeps the existing seal round.
        let kind = if self.plan.gu.is_some() {
            KIND_SEED
        } else {
            KIND_SEAL
        };
        let prefix = message.prefix(kind, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.gu_root = Some(*root);
        if let Some(gu) = &self.plan.gu {
            self.gu_registry.receive(
                gu,
                &GuNotice {
                    root: *root,
                    oracle: gu.oracle,
                    tree_number: gu.tree.tree_number,
                    part: gu.part,
                },
            )?;
        }
        self.stage = Stage::Gu;
        Ok(())
    }

    /// Draw the verifier's shared seed set from the session chain after the
    /// received G_u root. The seed carries the maximum sample-cube dimension
    /// and G_u's answer count; the caller cannot supply or replace it. This is
    /// the single-G_u interface; a multi-oracle SEND message is not
    /// implemented.
    pub fn shared_seeds(&mut self) -> Result<()> {
        if self.stage != Stage::Gu {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?;
        let bits = gu.sample_bits()? as usize;
        let answers = gu.u.answers as usize;
        let mut seeds = Vec::new();
        seeds
            .try_reserve_exact(answers)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..answers {
            let mut z = Vec::new();
            z.try_reserve_exact(bits)
                .map_err(|_| Error::ResourceLimit)?;
            for _ in 0..bits {
                z.push(self.chain.draw_e()?);
            }
            let beta = self.chain.draw_e_nonzero()?;
            seeds.push(SampleSeed { z, beta });
        }
        self.gu_seeds = seeds;
        self.gu_answers.clear();
        self.gu_answered = 0;
        self.stage = Stage::GuAnswers;
        Ok(())
    }

    /// Receive the next sample answer in the fixed single-G_u answer order.
    /// The answer is encoded into the answer round with its canonical field
    /// words; after the last answer the round is closed and its state feeds
    /// the later column draws. A value swap is not necessarily visible at this
    /// typed interface; the fixed seed and later relation check decide it.
    ///
    /// The interface takes typed field values. A proof-byte decoder with
    /// truncation/non-canonical checks is not connected here, so that byte
    /// boundary remains an integration gate and is not claimed as tested.
    pub fn receive_gu_answer(&mut self, target: Fp3) -> Result<()> {
        if self.stage != Stage::GuAnswers {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        if self.gu_answered >= gu.u.answers as usize {
            return Err(Error::InvalidSessionState);
        }
        let row = sample_row(
            &gu,
            &self.gu_registry,
            &self.gu_seeds,
            self.gu_answered as u32,
            target,
        )?;
        self.gu_rows.push(row);
        self.gu_answers.push(target);
        self.gu_answered += 1;
        if self.gu_answered == gu.u.answers as usize {
            let mut message = self.chain.message();
            for answer in &self.gu_answers {
                message.push_fp3(answer);
            }
            let prefix = message.prefix(KIND_EMPTY, 0)?;
            if !self.chain.step(&prefix, 0)? {
                return Err(Error::InvalidProof);
            }
            self.stage = Stage::GuRecord;
        }
        Ok(())
    }

    /// Join the trusted group and the record rows after every sample answer,
    /// using the same gamma as the row-point weights and the bound index
    /// name's record seeds. The group join checks the trusted group/code and
    /// the registered G_u shape, and every pending row must belong to the
    /// joined oracle and group.
    pub fn join_gu_record(&mut self) -> Result<()> {
        if self.stage != Stage::GuRecord {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let group = self
            .plan
            .statement
            .groups()
            .get(gu.group)
            .ok_or(Error::StatementMismatch)?;
        let mut membership = GuMembership::new(group, &gu, &self.gu_registry)?;
        membership.join(&gu, &self.gu_registry)?;
        for row in &self.gu_rows {
            membership.check_row(row)?;
        }
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let name = index_name(
            &self.plan.reader.root,
            &self.plan.reader.shape,
            &self.plan.reader.configuration,
        );
        let answers = self.plan.reader.shape.answers as usize;
        let seeds = self.plan.seeds()?;
        if seeds.len() < answers {
            return Err(Error::ShapeMismatch);
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(answers)
            .map_err(|_| Error::ResourceLimit)?;
        for answer in 0..answers {
            let row = record_row(
                &gu,
                &self.gu_registry,
                &self.plan.reader.record,
                &name,
                &seeds[..answers],
                &eq_gamma,
                answer as u32,
            )?;
            membership.check_row(&row)?;
            rows.push(row);
        }
        self.gu_rows.extend(rows);
        self.gu_membership = Some(membership);
        self.stage = Stage::GuJoined;
        Ok(())
    }

    /// Add column rows from an explicit fixture column map before any read.
    /// The fixture is not authenticated by a reader opening and the rows stay
    /// pending; this path is only for isolated evaluation and is not accepted
    /// by the FOLD gate. A caller with verified reads must use
    /// `join_gu_read_columns` after `read_queries`.
    pub fn join_gu_columns(&mut self, fixture_columns: &[(u64, Vec<Fp3>)]) -> Result<()> {
        if self.stage != Stage::Queried {
            return Err(Error::InvalidSessionState);
        }
        if self.gu_columns_source.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let membership = self
            .gu_membership
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        membership.require(&gu, &self.gu_registry)?;
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let columns = column_plan_columns(
            &gu,
            self.plan.column_draws,
            self.plan.reader.shape.code.log_n,
            &self.queries,
            fixture_columns,
        )?;
        if gu_layout_fault(&gu).is_some() {
            return Err(Error::InvalidConfig);
        }
        let mut targets = Vec::with_capacity(columns.len());
        for (_, column) in &columns {
            targets.push(column_target(&gu, column, &eq_gamma)?);
        }
        self.gu_fixture_columns = Some(fixture_columns.to_vec());
        self.gu_column_targets = targets;
        self.gu_columns_source = Some(GuColumnSource::Fixture);
        Ok(())
    }

    /// Add column rows from the columns that `read_queries` verified against
    /// the trusted reader root for this session's draws. The unique loop below
    /// builds only the physical position→column map; the complete `queries`
    /// sequence is then passed to `column_plan_rows`, which emits one logical
    /// relation row per draw, repeats included. Equal openings for a repeated
    /// draw are checked in `read_queries` before its unique position is sent to
    /// `VerifierReads`; that store's `(root, position)` record is for values
    /// read again across separate reads, not a second handling of this same
    /// repeat. Provider hiding accounting is separate and is not reduced here.
    pub fn join_gu_read_columns(&mut self) -> Result<()> {
        if self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if self.gu_columns_source.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let membership = self
            .gu_membership
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        membership.require(&gu, &self.gu_registry)?;
        let read_columns = self
            .gu_read_columns
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        if read_columns.len() != self.queries.len() {
            return Err(Error::ShapeMismatch);
        }
        let rows_count = usize::try_from(self.plan.reader.shape.rows())
            .map_err(|_| Error::ResourceLimit)?;
        let mut columns = Vec::new();
        for position in unique_sorted(&self.queries) {
            let mut column: Option<&Vec<Fp>> = None;
            for (query, read) in self.queries.iter().zip(read_columns) {
                if *query == position {
                    if read.len() != rows_count {
                        return Err(Error::ShapeMismatch);
                    }
                    match column {
                        Some(previous) if previous != read => {
                            return Err(Error::StatementMismatch);
                        }
                        Some(_) => {}
                        None => column = Some(read),
                    }
                }
            }
            let column = column.ok_or(Error::StatementMismatch)?;
            let values = column
                .iter()
                .map(|value| Fp3::from_base(*value))
                .collect();
            columns.push((position, values));
        }
        let gamma = self.gamma.clone().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(&gamma, self.plan.reader.shape.v)?;
        let columns = column_plan_columns(
            &gu,
            self.plan.column_draws,
            self.plan.reader.shape.code.log_n,
            &self.queries,
            &columns,
        )?;
        if gu_layout_fault(&gu).is_some() {
            return Err(Error::InvalidConfig);
        }
        let mut targets = Vec::with_capacity(columns.len());
        for (_, column) in &columns {
            targets.push(column_target(&gu, column, &eq_gamma)?);
        }
        self.gu_column_targets = targets;
        self.gu_columns_source = Some(GuColumnSource::Verified);
        Ok(())
    }

    /// SEND of Omega: bind the received root on the shared chain after the
    /// table columns were verified. Omega is a static member of the FOLD
    /// group; its sample answer round follows.
    pub fn send_omega(&mut self, root: &Digest) -> Result<()> {
        if self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if self.gu_columns_source != Some(GuColumnSource::Verified)
            || self.gu_membership.is_none()
        {
            return Err(Error::InvalidSessionState);
        }
        let omega = self
            .plan
            .omega
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        if *root == Digest::ZERO {
            return Err(Error::InvalidProof);
        }
        let mut message = self.chain.message();
        for chunk in root.0.chunks(8) {
            let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
            message.push_u64(word);
        }
        let prefix = message.prefix(KIND_SEED, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.omega_root = Some(*root);
        self.omega_seeds.clear();
        self.omega_answers.clear();
        self.omega_answered = 0;
        self.omega_rows.clear();
        self.omega_joined = false;
        let _ = omega;
        self.stage = Stage::Omega;
        Ok(())
    }

    /// Draw Omega's own sample seeds after its root. The seed width is the
    /// oracle's cube width; the caller cannot supply or replace it.
    pub fn omega_seeds(&mut self) -> Result<Vec<SampleSeed>> {
        if self.stage != Stage::Omega {
            return Err(Error::InvalidSessionState);
        }
        let omega = self
            .plan
            .omega
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        let bits = sample_cube_of(&omega.u)?.bits as usize;
        let answers = omega.u.answers as usize;
        let mut seeds = Vec::with_capacity(answers);
        for _ in 0..answers {
            let mut z = Vec::with_capacity(bits);
            for _ in 0..bits {
                z.push(self.chain.draw_e()?);
            }
            let beta = self.chain.draw_e_nonzero()?;
            seeds.push(SampleSeed { z, beta });
        }
        self.omega_seeds = seeds.clone();
        self.omega_answers.clear();
        self.omega_answered = 0;
        self.stage = Stage::OmegaAnswers;
        Ok(seeds)
    }

    /// Receive the next Omega sample answer in the fixed order. The row uses
    /// the ordinary-oracle sample relation; the answer round closes after the
    /// last answer.
    pub fn receive_omega_answer(&mut self, target: Fp3) -> Result<()> {
        if self.stage != Stage::OmegaAnswers {
            return Err(Error::InvalidSessionState);
        }
        let omega = self
            .plan
            .omega
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        if self.omega_answered >= omega.u.answers as usize {
            return Err(Error::InvalidSessionState);
        }
        let row = oracle_sample_row(
            omega.oracle,
            omega.group,
            &omega.u,
            &self.omega_seeds,
            self.omega_answered as u32,
            target,
        )?;
        self.omega_rows.push(row);
        self.omega_answers.push(target);
        self.omega_answered += 1;
        if self.omega_answered == omega.u.answers as usize {
            let mut message = self.chain.message();
            for answer in &self.omega_answers {
                message.push_fp3(answer);
            }
            let prefix = message.prefix(KIND_EMPTY, 0)?;
            if !self.chain.step(&prefix, 0)? {
                return Err(Error::InvalidProof);
            }
            self.stage = Stage::OmegaJoined;
        }
        Ok(())
    }

    /// Join Omega's sample rows into the FOLD group. The root and the static
    /// group membership are already fixed; every row must name Omega and the
    /// FOLD group.
    pub fn join_omega(&mut self) -> Result<()> {
        if self.stage != Stage::OmegaJoined || self.omega_joined {
            return Err(Error::InvalidSessionState);
        }
        let omega = self
            .plan
            .omega
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        let membership = self
            .gu_membership
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        if omega.group != self.plan.group_id
            || membership.group != omega.group
            || !membership.members.contains(&omega.oracle)
            || self.omega_root.is_none()
            || self.omega_rows.len() != omega.u.answers as usize
            || self
                .omega_rows
                .iter()
                .any(|row| row.oracle != omega.oracle || row.group != omega.group)
        {
            return Err(Error::StatementMismatch);
        }
        self.omega_joined = true;
        Ok(())
    }

    /// Draw the row point zeta after Omega was joined. Its width is the
    /// matrix row bits; the caller cannot supply it.
    pub fn draw_zeta(&mut self) -> Result<Vec<Fp3>> {
        if !self.omega_joined || self.zeta.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let omega = self
            .plan
            .omega
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        if omega.fault().is_some() {
            return Err(Error::InvalidConfig);
        }
        let k_row = omega.matrix_rows.trailing_zeros();
        let prefix = self.chain.message().prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut zeta = Vec::with_capacity(k_row as usize);
        for _ in 0..k_row {
            zeta.push(self.chain.draw_e()?);
        }
        self.zeta = Some(zeta.clone());
        self.stage = Stage::ZetaDrawn;
        Ok(zeta)
    }

    pub fn zeta(&self) -> Option<&[Fp3]> {
        self.zeta.as_deref()
    }

    /// Draw the BAT group challenge after the row point. The challenge is
    /// absorbed on the shared chain and the same value is used to combine the
    /// group's frozen rows.
    pub fn draw_bat_challenge(&mut self) -> Result<Fp3> {
        if self.stage != Stage::ZetaDrawn {
            return Err(Error::InvalidSessionState);
        }
        let prefix = self.chain.message().prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let x = self.chain.draw_e()?;
        self.bat_x = Some(x);
        self.stage = Stage::BatDrawn;
        Ok(x)
    }

    pub fn bat_challenge(&self) -> Option<Fp3> {
        self.bat_x
    }

    pub fn pending_gu_rows(&self) -> &[GuRelationRow] {
        &self.gu_rows
    }

    /// The logical column draws whose targets are pending in this session.
    /// Column coefficient rows are built one at a time on demand from the
    /// verified columns instead of being retained as dense tables.
    pub fn pending_column_count(&self) -> usize {
        self.gu_column_targets.len()
    }

    pub fn column_targets(&self) -> &[Fp3] {
        &self.gu_column_targets
    }

    /// Build one pending column relation row from its source. The caller gets
    /// one row at a time, so a large draw set is never all materialized.
    pub fn column_relation_row(&self, draw: usize) -> Result<GuRelationRow> {
        let source = self
            .gu_columns_source
            .ok_or(Error::InvalidSessionState)?;
        let gu = self.plan.gu.as_ref().ok_or(Error::InvalidSessionState)?;
        let position = *self.queries.get(draw).ok_or(Error::InvalidIndex)?;
        let gamma = self.gamma.as_ref().ok_or(Error::InvalidSessionState)?;
        let eq_gamma = foldeq_weights(gamma, self.plan.reader.shape.v)?;
        match source {
            GuColumnSource::Fixture => {
                let fixture = self
                    .gu_fixture_columns
                    .as_ref()
                    .ok_or(Error::InvalidSessionState)?;
                let (_, column) = fixture
                    .iter()
                    .find(|(at, _)| *at == position)
                    .ok_or(Error::StatementMismatch)?;
                column_row(gu, &self.gu_registry, position, column, &eq_gamma)
            }
            GuColumnSource::Verified => {
                let reads = self
                    .gu_read_columns
                    .as_ref()
                    .ok_or(Error::InvalidSessionState)?;
                let read = reads.get(draw).ok_or(Error::InvalidIndex)?;
                let column: Vec<Fp3> = read.iter().map(|value| Fp3::from_base(*value)).collect();
                column_row(gu, &self.gu_registry, position, &column, &eq_gamma)
            }
        }
    }

    pub fn omega_rows(&self) -> &[GuRelationRow] {
        &self.omega_rows
    }

    pub fn draw_queries(&mut self) -> Result<()> {
        self.draw_queries_with_work(0, Some(0)).map(|_| ())
    }

    /// Draws the column positions with `work_bits` of proof of work before the
    /// verifier's message. The prover passes `None` and searches; the verifier
    /// passes the nonce carried by the proof. A failed work test does not
    /// advance the chain. Returns the nonce (zero when no work is asked).
    pub fn draw_queries_with_work(
        &mut self,
        work_bits: u8,
        nonce: Option<u64>,
    ) -> Result<u64> {
        let expected = if self.plan.gu.is_some() {
            Stage::GuJoined
        } else {
            Stage::Gu
        };
        if self.stage != expected || work_bits > MAX_WORK_BITS {
            return Err(Error::InvalidSessionState);
        }
        let prefix = self.chain.message().prefix(KIND_POSITIONS, work_bits)?;
        let nonce = match nonce {
            Some(value) => {
                if !verify_nonce(&prefix, value, work_bits)? {
                    return Err(Error::InvalidProof);
                }
                value
            }
            None => search_nonce_for_plan(&prefix)?,
        };
        if !self.chain.step(&prefix, nonce)? {
            return Err(Error::InvalidProof);
        }
        self.queries = self.plan.draw_queries(&mut self.chain)?;
        self.stage = Stage::Queried;
        Ok(nonce)
    }

    /// Verify one opening per unique physical position and map the verified
    /// columns back to the logical draw sequence. The caller still supplies one
    /// opening per logical draw; repeats must carry the same opening. Only the
    /// verification and the read ledger deduplicate to unique physical
    /// positions. This does not claim that an online proof carries one opening
    /// per position or that proof size is reduced; the real opening list
    /// encoding and provision remain a production integration gate. The read
    /// ledger charges the actual unique openings, while the plan's column count
    /// remains the logical sampling count. Hiding accounting is a separate
    /// provider-side ledger and is not reduced by this deduplication.
    pub fn read_queries(&mut self, openings: &[ColumnOpening]) -> Result<Vec<Vec<Fp>>> {
        if self.stage != Stage::Queried && self.stage != Stage::Read {
            return Err(Error::InvalidSessionState);
        }
        if openings.len() != self.queries.len() {
            return Err(Error::ShapeMismatch);
        }
        let Self {
            plan,
            reads,
            queries,
            stage,
            gu_read_columns,
            ..
        } = self;
        let mut unique: Vec<(u64, ColumnOpening)> = Vec::new();
        for (position, opening) in queries.iter().zip(openings) {
            if opening.position != *position {
                return Err(Error::InvalidIndex);
            }
            match unique.iter().find(|(at, _)| at == position) {
                Some((_, previous)) if previous != opening => {
                    return Err(Error::StatementMismatch);
                }
                Some(_) => {}
                None => unique.push((*position, opening.clone())),
            }
        }
        unique.sort_by_key(|(position, _)| *position);
        let positions: Vec<u64> = unique.iter().map(|(position, _)| *position).collect();
        let unique_openings: Vec<ColumnOpening> =
            unique.into_iter().map(|(_, opening)| opening).collect();
        let unique_columns = reads.read(&plan.reader, &positions, &unique_openings)?;
        let mut by_position = std::collections::HashMap::new();
        for (position, column) in positions.into_iter().zip(unique_columns) {
            by_position.insert(position, column);
        }
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(queries.len())
            .map_err(|_| Error::ResourceLimit)?;
        for position in queries {
            columns.push(
                by_position
                    .get(position)
                    .cloned()
                    .ok_or(Error::StatementMismatch)?,
            );
        }
        if plan.gu.is_some() {
            *gu_read_columns = Some(columns.clone());
        }
        *stage = Stage::Read;
        Ok(columns)
    }

    /// Verify the prover's public FOLD start message and bind it in the start
    /// round; the verifier receives no coefficient, tail or coin. With a G_u
    /// shape the session overlays the joined membership onto the trusted group
    /// view consumed by the FOLD check, and requires the FOLD target group to
    /// be the joined group and the column rows to come from verified reads.
    pub fn start_fold(&mut self, message: &FoldStartMessage) -> Result<()> {
        if self.stage != Stage::Read
            && self.stage != Stage::ZetaDrawn
            && self.stage != Stage::BatDrawn
        {
            return Err(Error::InvalidSessionState);
        }
        if self.plan.gu.is_some() {
            if self.gu_membership.is_none() {
                return Err(Error::InvalidSessionState);
            }
            if self.gu_columns_source != Some(GuColumnSource::Verified) {
                return Err(Error::InvalidSessionState);
            }
            if self.fold_row.is_none() {
                return Err(Error::InvalidSessionState);
            }
        }
        if message.tree_number != self.plan.mask_tree_number
            || message.claimed_group != self.plan.group_id
        {
            return Err(Error::StatementMismatch);
        }
        let gu_groups = if let Some(gu) = self.plan.gu.as_ref() {
            let membership = self
                .gu_membership
                .as_ref()
                .ok_or(Error::InvalidSessionState)?;
            membership.require(gu, &self.gu_registry)?;
            if membership.group != self.plan.group_id {
                return Err(Error::StatementMismatch);
            }
            let mut groups = self.plan.statement.groups().clone();
            for group in &mut groups.groups {
                if group.group == membership.group {
                    group.members = membership.members.clone();
                }
            }
            Some(groups)
        } else {
            None
        };
        let oracles = self
            .plan
            .oracles
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        match &gu_groups {
            Some(groups) => fold_start_verify_with_groups(
                &self.plan.fold,
                message,
                &self.plan.statement,
                &oracles,
                groups,
            )?,
            None => fold_start_verify(&self.plan.fold, message, &self.plan.statement, &oracles)?,
        }
        let mut out = self.chain.message();
        out.push_u64(message.id);
        out.push_u64(message.claimed_group);
        out.push_u64(message.notice.mask_rows);
        out.push_u64(message.notice.mask_code.log_n as u64);
        out.push_u64(message.notice.mask_code.ell);
        out.push_u64(message.notice.mask_code.r);
        out.push_u64(message.notice.answers as u64);
        out.push_u64(message.notice.leaf_class as u64);
        for chunk in message.root.0.chunks(8) {
            let word = u64::from_le_bytes(chunk.try_into().map_err(|_| Error::InvalidEncoding)?);
            out.push_u64(word);
        }
        out.push_fp3(&message.sum);
        let prefix = out.prefix(KIND_EPSILON, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        self.epsilon = Some(self.chain.draw_e()?);
        self.start = Some(message.clone());
        self.stage = Stage::FoldStarted;
        Ok(())
    }

    /// Fix the FOLD input row before any challenge is drawn. The row is an
    /// isolated fixture, but its complete coefficients and target are held by
    /// the session from this point: `start_fold` consumes it and no later call
    /// can replace it. The group/layout/membership checks use the trusted plan
    /// and the session's fixed membership, never a proof-supplied row.
    pub fn fold_bind_row(&mut self, row: FoldInputRow) -> Result<()> {
        if self.stage != Stage::Read
            && self.stage != Stage::ZetaDrawn
            && self.stage != Stage::BatDrawn
        {
            return Err(Error::InvalidSessionState);
        }
        if self.fold_row.is_some() {
            return Err(Error::InvalidSessionState);
        }
        self.check_fold_row(&row, None)?;
        self.fold_row = Some(row);
        Ok(())
    }

    fn check_fold_row(&self, row: &FoldInputRow, claimed_group: Option<u64>) -> Result<()> {
        let layout = self
            .plan
            .fold_layout
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        row.check(&self.plan.fold, layout)?;
        if row.group != self.plan.group_id || layout.group != self.plan.group_id {
            return Err(Error::StatementMismatch);
        }
        if let Some(claimed) = claimed_group {
            if row.group != claimed {
                return Err(Error::StatementMismatch);
            }
        }
        let expected_members = match self.plan.gu.as_ref() {
            Some(gu) => {
                let membership = self
                    .gu_membership
                    .as_ref()
                    .ok_or(Error::InvalidSessionState)?;
                membership.require(gu, &self.gu_registry)?;
                if membership.group != row.group {
                    return Err(Error::StatementMismatch);
                }
                membership.members.clone()
            }
            None => self
                .plan
                .statement
                .groups()
                .get(self.plan.group_id)
                .ok_or(Error::StatementMismatch)?
                .members
                .clone(),
        };
        if row.members != expected_members
            || layout
                .members
                .iter()
                .map(|member| member.oracle)
                .ne(expected_members.iter().copied())
        {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    /// Start the FOLD rounds from the already-fixed input row. The row was
    /// bound before epsilon; this entry cannot accept or replace it. The
    /// verifier's running claim begins at the received hiding sum, epsilon and
    /// the fixed target.
    pub fn fold_rounds_start(&mut self) -> Result<()> {
        if self.stage != Stage::FoldStarted {
            return Err(Error::InvalidSessionState);
        }
        let start = self.start.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let eps = self.epsilon.ok_or(Error::InvalidSessionState)?;
        let row = self
            .fold_row
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        self.check_fold_row(&row, Some(start.claimed_group))?;
        self.fold_claim = Some(FoldRoundVerifier::new(start.sum, eps, row.target));
        self.fold_gamma.clear();
        self.fold_rounds = 0;
        self.fold_result = None;
        self.stage = Stage::FoldRounds;
        Ok(())
    }

    /// Receive one round's constant and quadratic coefficients. The complete
    /// two-value message is absorbed before the same chain draws this round's
    /// challenge; the verifier then updates the running claim from the message.
    /// The protocol plan's per-round error-parameter check is not connected in
    /// this isolated path, and the typed message is not a canonical proof-byte
    /// decoder.
    pub fn receive_fold_round(&mut self, c0: Fp3, c2: Fp3) -> Result<()> {
        if self.stage != Stage::FoldRounds {
            return Err(Error::InvalidSessionState);
        }
        if self.fold_rounds >= self.plan.fold.k {
            return Err(Error::InvalidSessionState);
        }
        let mut message = self.chain.message();
        message.push_fp3(&c0);
        message.push_fp3(&c2);
        let prefix = message.prefix(KIND_ALGEBRAIC, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let gamma = self.chain.draw_e()?;
        let claim = self
            .fold_claim
            .as_mut()
            .ok_or(Error::InvalidSessionState)?;
        claim.round(c0, c2, gamma);
        self.fold_gamma.push(gamma);
        self.fold_rounds += 1;
        Ok(())
    }

    /// Finish the FOLD rounds and store the final constraint. The output is
    /// the candidate for the next protocol stage, not an acceptance: the
    /// IOPP/CS continuation is not connected and fails closed.
    pub fn fold_finish(&mut self) -> Result<FoldOutput> {
        if self.stage != Stage::FoldRounds {
            return Err(Error::InvalidSessionState);
        }
        if self.fold_rounds != self.plan.fold.k {
            return Err(Error::InvalidSessionState);
        }
        let start = self.start.as_ref().ok_or(Error::InvalidSessionState)?.clone();
        let row = self
            .fold_row
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .clone();
        let claim = self
            .fold_claim
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .claim();
        let layout = self
            .plan
            .fold_layout
            .as_ref()
            .ok_or(Error::InvalidSessionState)?;
        let output = fold_output(
            &self.plan.fold,
            layout,
            &row,
            start.id,
            start.root,
            start.tree_number,
            self.epsilon.ok_or(Error::InvalidSessionState)?,
            start.sum,
            &self.fold_gamma,
            claim,
        )?;
        self.fold_result = Some(output.clone());
        self.stage = Stage::FoldFinished;
        Ok(output)
    }

    /// The stored final FOLD constraint, if the rounds completed.
    pub fn fold_output(&self) -> Option<&FoldOutput> {
        self.fold_result.as_ref()
    }

    /// Transfers chain ownership to the isolated fixture BASE stage after the
    /// FOLD rounds finished. This entry is for one fixture continuation only:
    /// it requires a stored FOLD output and consumes the session, so the FOLD
    /// handle cannot advance independently. It is not a production acceptance
    /// path; `fold_next_stage` remains fail-closed.
    pub fn take_chain_for_base(self) -> Result<Chain> {
        if self.stage != Stage::FoldFinished || self.fold_result.is_none() {
            return Err(Error::InvalidSessionState);
        }
        Ok(self.chain)
    }

    /// The next stage after FOLD is not connected. This method records the
    /// boundary explicitly and always fails closed rather than accepting the
    /// output as a complete proof.
    pub fn fold_next_stage(&self) -> Result<()> {
        self.fold_result
            .as_ref()
            .ok_or(Error::InvalidSessionState)?
            .next_stage()
    }

    /// The challenges drawn by the verified rounds, for tests and inspection.
    pub fn fold_gamma(&self) -> &[Fp3] {
        &self.fold_gamma
    }

    /// Draw the mask opening positions from the verifier's own transcript.
    /// The plan supplies only the query count; the positions are not the
    /// plan's list and are not supplied by the caller or the proof. With a G_u
    /// shape this legacy path is unavailable; the S1 flow proceeds through the
    /// fixed FOLD rounds. The opening/authentication and plan error-parameter
    /// checks remain unconnected.
    pub fn draw_mask_queries(&mut self) -> Result<()> {
        if self.stage != Stage::FoldStarted {
            return Err(Error::InvalidSessionState);
        }
        // With a G_u shape the S1 path continues through the fixed FOLD rounds
        // and their final constraint; the legacy mask-query path is not an
        // alternative and remains unavailable.
        if self.plan.gu.is_some() {
            return Err(Error::InvalidSessionState);
        }
        let tree = self
            .plan
            .statement
            .mask_tree(self.plan.mask_tree_number)
            .ok_or(Error::StatementMismatch)?;
        let prefix = self.chain.message().prefix(KIND_POSITIONS, 0)?;
        if !self.chain.step(&prefix, 0)? {
            return Err(Error::InvalidProof);
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(self.plan.mask_query_count)
            .map_err(|_| Error::ResourceLimit)?;
        for _ in 0..self.plan.mask_query_count {
            positions.push(self.chain.draw_position(tree.log_n)?);
        }
        self.mask_queries = unique_sorted(&positions);
        self.stage = Stage::MaskQueried;
        Ok(())
    }

    /// Check the mask opening at the positions the verifier's transcript drew.
    /// The plan's count is a different source from the drawn list; the
    /// low-level check accepts at most that many positions and does not prove
    /// the list's source or completeness. The list comes from this session's
    /// own chain in the harness, which is not connected to production and does
    /// not claim a complete round, proximity, table test or hiding
    /// closure. With a zero plan count there is nothing to open: this returns
    /// `NoOpening`, does not mark the mask tree as opened, and leaves the stage
    /// at `MaskQueried`; it is an explicit no-op, not a successful mask
    /// authentication. Whether the block permits zero queries is a
    /// block condition.
    pub fn read_mask(&mut self, openings: &RuntimeTreeOpening) -> Result<MaskAuthentication> {
        if self.stage != Stage::MaskQueried {
            return Err(Error::InvalidSessionState);
        }
        let message = self.start.as_ref().ok_or(Error::InvalidSessionState)?;
        let tree = self
            .plan
            .statement
            .mask_tree(self.plan.mask_tree_number)
            .ok_or(Error::StatementMismatch)?;
        let mut oracles = self
            .plan
            .oracles
            .lock()
            .map_err(|_| Error::InvalidSessionState)?;
        let outcome = oracles.authenticate_mask(
            tree,
            message.id,
            self.plan.mask_query_count,
            &self.mask_queries,
            openings,
        )?;
        drop(oracles);
        match outcome {
            MaskAuthentication::NoOpening => Ok(MaskAuthentication::NoOpening),
            MaskAuthentication::Verified => {
                self.stage = Stage::MaskOpened;
                Ok(MaskAuthentication::Verified)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::request::code::{build_code_plan, CodeParams};
    use crate::pcs::request::field::{Fp, Fp3, MODULUS};
    use crate::pcs::request::fold::{
        fold_mask_make, fold_mask_make_tail, fold_mask_notice, fold_start_prove,
        FoldInputRow, FoldKeptForm, FoldMask, FoldMemberLayout, FoldOutput,
        FoldProverSession, FoldRoundEnd, FoldShape,
        MaskOracle, OracleMetadata, RuntimeOracles, VerifiedGroup, VerifiedGroups,
        VerifiedStatement,
    };
    use crate::pcs::request::binding::{BindingParameters, BindingStatement};
    use crate::pcs::request::params::{
        RequestParams, LEAF_SALT_MIN_BYTES, REQUEST_CHAIN_LABEL,
    };
    use crate::pcs::request::gu::{GuNotice, GuOracleShape, GuRegistry, GuShape};
    use crate::pcs::request::read::{unique_sorted, HidingState, ProverReads};
    use crate::pcs::request::relation::{gu_table, GuRelationKind, GuWitness};
    use crate::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
    use crate::pcs::request::runtime_tree::{verify_mask_opening, MaskTreePlan};
    use crate::pcs::request::table::{
        test_commit_table, IndexReader, IndexRecord, IndexShape, TableDistances,
    };
    use crate::primitives::transcript::{hash_labeled, tag_word, TEST_COINS};
    use zkie_core::common::field::PrimeCharacteristicRing;
    use zkie_core::proof_header::PROOF_VERSION_WORD;

    fn binding_parameters(shape: &IndexShape) -> BindingParameters {
        let request_parameters =
            RequestParams::default_with_code(shape.code.log_n, shape.code.ell, shape.code.r)
                .unwrap();
        BindingParameters::new(
            PROOF_VERSION_WORD,
            Digest([0x61; 32]),
            &request_parameters,
            Digest([0x62; 32]),
        )
        .unwrap()
    }

    fn binding_with(
        reader: &IndexReader,
        parameters: &BindingParameters,
        case_id: &[u8],
        request_id: Digest,
    ) -> FoldeqBinding {
        let statement = BindingStatement::new(
            parameters.version_word,
            parameters.protocol_identity,
            case_id.to_vec(),
            request_id,
            reader_table_identity(reader).unwrap(),
        )
        .unwrap();
        FoldeqBinding::new(parameters, statement, reader).unwrap()
    }

    fn binding_for(reader: &IndexReader) -> FoldeqBinding {
        let parameters = binding_parameters(&reader.shape);
        binding_with(reader, &parameters, b"case", Digest([0x63; 32]))
    }

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

    fn committed() -> crate::pcs::request::table::TestTable {
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
        let configuration = binding_parameters(&shape).digest().unwrap();
        test_commit_table(shape, configuration, &messages, &tails, &salts).unwrap()
    }

    fn fold_shape() -> FoldShape {
        FoldShape {
            code: build_code_plan(3, 4, 1).unwrap(),
            mask_code: build_code_plan(3, 4, 1).unwrap(),
            k: 2,
            h: 1,
            mask_leaf_class: 3,
        }
    }

    fn mask_tree() -> MaskTreePlan {
        let fold = fold_shape();
        MaskTreePlan {
            tree_number: 5,
            leaf_class: fold.mask_leaf_class,
            log_n: fold.mask_code.log_n,
            mask_rows: fold.mask_rows().unwrap(),
            salt_bytes: LEAF_SALT_MIN_BYTES,
            run: Digest([2; 32]),
            configuration: Digest([7; 32]),
        }
    }

    fn mask_salts(plan: &MaskTreePlan) -> Vec<Vec<u8>> {
        (0..plan.columns() as usize)
            .map(|position| vec![(position as u8).wrapping_add(1); plan.salt_bytes])
            .collect()
    }

    fn gu_shape() -> GuShape {
        GuShape {
            oracle: 10,
            group: 1,
            tree: RuntimeTreePlan {
                tree_number: 6,
                leaf_class: 3,
                log_n: 3,
                log_n_zk: 0,
                parts: vec![RuntimeTreePart {
                    main_symbols: 6,
                    mask_symbols: 0,
                }],
                salt_bytes: LEAF_SALT_MIN_BYTES,
                run: Digest([9; 32]),
                configuration: Digest([7; 32]),
            },
            part: 0,
            rows_high: false,
            u: GuOracleShape {
                code: build_code_plan(3, 6, 1).unwrap(),
                main_rows: 2,
                main_deg: 3,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: shape(),
        }
    }

    fn fold_layout_for(plan: &FoldeqPlan, oracle: u64) -> FoldGroupLayout {
        let mut oracles = plan
            .statement
            .groups()
            .get(plan.group_id)
            .unwrap()
            .members
            .clone();
        oracles.push(oracle);
        oracles.sort_unstable();
        oracles.dedup();
        let gu_rows = plan.gu.as_ref().map(|gu| gu.u.main_rows).unwrap_or(1);
        let members = oracles
            .into_iter()
            .map(|id| FoldMemberLayout {
                oracle: id,
                main_rows: if id == oracle {
                    gu_rows
                } else if id == 2 {
                    2
                } else {
                    1
                },
                mask_rows: 0,
                kept: id == 1 || id == 2,
            })
            .collect();
        FoldGroupLayout {
            group: plan.group_id,
            k: plan.fold.k,
            h: plan.fold.h,
            members,
        }
    }

    fn gu_plan_with(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
        oracle: u64,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        let (mut plan, table) = plan_fixture_with_group(group_code, members);
        let mut gu = gu_shape();
        gu.oracle = oracle;
        plan.gu = Some(gu);
        plan.fold_layout = Some(fold_layout_for(&plan, oracle));
        (plan, table)
    }

    fn gu_plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        gu_plan_with(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2], 10)
    }
    fn gu_plan_two_groups(
        gu_group: u64,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        let code = Some(build_code_plan(3, 6, 1).unwrap());
        let (mut plan, table) = plan_fixture_with_groups(vec![
            VerifiedGroup {
                group: 1,
                members: vec![1, 2],
                fresh: Some(9),
                code,
                mask_code: None,
            },
            VerifiedGroup {
                group: 2,
                members: vec![3, 4],
                fresh: Some(8),
                code,
                mask_code: None,
            },
        ]);
        let mut gu = gu_shape();
        gu.group = gu_group;
        let oracle = gu.oracle;
        plan.gu = Some(gu);
        plan.fold_layout = Some(fold_layout_for(&plan, oracle));
        (plan, table)
    }

    fn gu_plan_fresh_collision() -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        let (mut plan, table) =
            plan_fixture_with_group_fresh(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2], Some(10));
        plan.gu = Some(gu_shape());
        plan.fold_layout = Some(fold_layout_for(&plan, 10));
        (plan, table)
    }

    fn gu_plan_with_answers(
        answers: u32,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        let (mut plan, table) = gu_plan_fixture();
        plan.gu.as_mut().unwrap().u.answers = answers;
        (plan, table)
    }

    fn open_gu_plan_from(plan: FoldeqPlan) -> FoldeqSession {
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan, binder, chain).unwrap();
        session
            .bind_first_query(&session.plan().first_query())
            .unwrap();
        session.seeds().unwrap();
        session.draw_gamma().unwrap();
        session
    }

    fn open_gu_plan() -> (FoldeqSession, crate::pcs::request::table::TestTable) {
        let (plan, table) = gu_plan_fixture();
        (open_gu_plan_from(plan), table)
    }

    fn open_gu_session() -> (FoldeqSession, crate::pcs::request::table::TestTable) {
        let (mut session, table) = open_gu_plan();
        session.send_gu(&Digest([4; 32])).unwrap();
        (session, table)
    }

    fn open_gu_with_root(
        root: Digest,
    ) -> (FoldeqSession, crate::pcs::request::table::TestTable) {
        let (mut session, table) = open_gu_plan();
        session.send_gu(&root).unwrap();
        (session, table)
    }

    fn plan_fixture_with_groups(
        groups: Vec<VerifiedGroup>,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        let table = committed();
        let tree = mask_tree();
        let statement = Arc::new(
            VerifiedStatement::new(
                VerifiedGroups { groups },
                vec![tree],
                vec![OracleMetadata {
                    id: 9,
                    group: 1,
                    tree_number: tree.tree_number,
                    leaf_class: tree.leaf_class,
                    mask_rows: tree.mask_rows,
                    mask_code: build_code_plan(3, 4, 1).unwrap(),
                }],
            )
            .unwrap(),
        );
        let plan = FoldeqPlan {
            reader: table.reader.clone(),
            binding: binding_for(&table.reader),
            edge: FoldeqEdgeBinding::NotAnEdge,
            fold: fold_shape(),
            statement,
            oracles: Arc::new(Mutex::new(RuntimeOracles::default())),
            mask_tree_number: tree.tree_number,
            group_id: 1,
            leaf_class: 3,
            seed_budget: table.reader.shape.answers as usize,
            ledger: Arc::new(Mutex::new(ReadLedger::new(2))),
            column_draws: 2,
            mask_query_count: 2,
            gamma_path: GammaPath::Lin { shape: LinShapeBinding { matrix_rows: 1, k_lo: 0, k_col: 2 } },
            gu: None,
            omega: None,
            fold_layout: None,
        };
        (plan, table)
    }

    fn plan_fixture_with_group_fresh(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
        fresh: Option<u64>,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        plan_fixture_with_groups(vec![VerifiedGroup {
            group: 1,
            members,
            fresh,
            code: group_code,
            mask_code: None,
        }])
    }

    fn plan_fixture_with_group(
        group_code: Option<CodeParams>,
        members: Vec<u64>,
    ) -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        plan_fixture_with_group_fresh(group_code, members, Some(9))
    }

    fn plan_fixture() -> (FoldeqPlan, crate::pcs::request::table::TestTable) {
        plan_fixture_with_group(Some(build_code_plan(3, 6, 1).unwrap()), vec![1, 2])
    }

    fn open_session() -> (FoldeqSession, crate::pcs::request::table::TestTable) {
        let (plan, table) = plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        (FoldeqSession::open(plan, binder, chain).unwrap(), table)
    }

    fn mask_material(
        plan: &FoldeqPlan,
    ) -> (TestCoins, FoldMask, crate::pcs::request::fold::FoldStartProver) {
        let mut source = TestCoins::new(Digest([9; 32]));
        let mut mask = fold_mask_make(&plan.fold, &mut source).unwrap();
        fold_mask_make_tail(&plan.fold, &mut mask, &mut source).unwrap();
        mask.sent_as(9).unwrap();
        let notice = fold_mask_notice(&plan.fold).unwrap();
        let oracle = MaskOracle::prover_view(9, 1, &notice);
        let tree = *plan.statement.mask_tree(plan.mask_tree_number).unwrap();
        let start = fold_start_prove(
            &plan.fold,
            &mut mask,
            &source,
            &oracle,
            plan.statement.groups(),
            tree,
            &mask_salts(&tree),
        )
        .unwrap();
        (source, mask, start)
    }

    fn mask_start(plan: &FoldeqPlan) -> crate::pcs::request::fold::FoldStartProver {
        mask_material(plan).2
    }

    fn fold_open_prestart() -> (
        FoldeqSession,
        crate::pcs::request::table::TestTable,
        crate::pcs::request::fold::FoldStartProver,
        FoldMask,
        TestCoins,
    ) {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        session.join_gu_read_columns().unwrap();
        let (source, mask, start) = mask_material(session.plan());
        register_mask_start(&session, &start.message);
        (session, table, start, mask, source)
    }

    fn fold_open_ready() -> (
        FoldeqSession,
        crate::pcs::request::table::TestTable,
        crate::pcs::request::fold::FoldStartProver,
        FoldMask,
        TestCoins,
    ) {
        let (mut session, table, start, mask, source) = fold_open_prestart();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        session.fold_bind_row(row).unwrap();
        session.start_fold(&start.message).unwrap();
        (session, table, start, mask, source)
    }

    fn fold_fixture_with_kept(
        plan: &FoldeqPlan,
        salt: u64,
        kept_salt: u64,
    ) -> (FoldInputRow, Vec<Fp3>, Vec<Fp3>, Vec<Vec<Fp3>>) {
        let shape = plan.fold;
        let rows = 1usize << shape.k;
        let mask_rows = 1usize << shape.h;
        let ell = shape.code.ell as usize;
        let mask_len = shape.mask_code.ell as usize;
        let coefficients: Vec<Fp3> = (0..rows * ell)
            .map(|index| Fp3::from_u64(salt * 1000 + index as u64 + 3))
            .collect();
        let main_messages: Vec<Fp3> = (0..rows * ell)
            .map(|index| Fp3::from_u64(salt * 1000 + index as u64 + 501))
            .collect();
        let mask_coefficients: Vec<Fp3> = (0..mask_rows * mask_len)
            .map(|index| Fp3::from_u64(salt * 1000 + index as u64 + 701))
            .collect();
        let mask_messages: Vec<Fp3> = (0..mask_rows * mask_len)
            .map(|index| Fp3::from_u64(salt * 1000 + index as u64 + 901))
            .collect();
        let kept_rows = plan
            .fold_layout
            .as_ref()
            .unwrap()
            .member(2)
            .unwrap()
            .main_rows as usize;
        let kept_form = FoldKeptForm {
            oracle: 2,
            coefficients: (0..kept_rows * ell)
                .map(|index| Fp3::from_u64(kept_salt * 1000 + 1201 + index as u64))
                .collect(),
        };
        let kept_message: Vec<Fp3> = (0..kept_rows * ell)
            .map(|index| Fp3::from_u64(kept_salt * 1000 + 1401 + index as u64))
            .collect();
        let kept_value = fold_dot(&kept_form.coefficients, &kept_message);
        let target = fold_dot(&coefficients, &main_messages)
            .add(&fold_dot(&mask_coefficients, &mask_messages))
            .add(&kept_value);
        let members = vec![1, 2, 10];
        let row = FoldInputRow {
            group: plan.group_id,
            k: shape.k,
            h: shape.h,
            members,
            coefficients,
            mask_coefficients,
            kept: vec![kept_form],
            target,
        };
        (row, main_messages, mask_messages, vec![kept_message])
    }

    fn fold_fixture(
        plan: &FoldeqPlan,
        salt: u64,
    ) -> (FoldInputRow, Vec<Fp3>, Vec<Fp3>, Vec<Vec<Fp3>>) {
        fold_fixture_with_kept(plan, salt, salt)
    }

    fn run_fold_instance(salt: u64) -> FoldRun {
        run_fold_instance_kept(salt, salt)
    }

    fn fold_dot(left: &[Fp3], right: &[Fp3]) -> Fp3 {
        left.iter()
            .zip(right)
            .fold(Fp3::ZERO, |sum, (a, b)| sum.add(&a.mul(b)))
    }

    struct FoldRun {
        messages: Vec<(Fp3, Fp3)>,
        qs: Vec<[Fp3; 3]>,
        output: FoldOutput,
        end: FoldRoundEnd,
        main_end: Vec<Fp3>,
        main_messages: Vec<Fp3>,
        mask_messages: Vec<Fp3>,
        row: FoldInputRow,
        kept_messages: Vec<Vec<Fp3>>,
        mask_coefficients: Vec<Fp3>,
        eps: Fp3,
        start: crate::pcs::request::fold::FoldStartMessage,
    }

    fn run_fold_instance_kept(salt: u64, kept_salt: u64) -> FoldRun {
        run_fold_instance_kept_rows(salt, kept_salt, None)
    }

    fn run_fold_instance_kept_rows(
        salt: u64,
        kept_salt: u64,
        kept_rows: Option<u64>,
    ) -> FoldRun {
        let (mut session, _table, start, mask, source) = fold_open_prestart();
        if let Some(rows) = kept_rows {
            let layout = session.plan.fold_layout.as_mut().unwrap();
            for member in &mut layout.members {
                if member.oracle == 2 {
                    member.main_rows = rows;
                }
            }
            assert!(session.plan.fault().is_none());
        }
        let (row, main_messages, mask_messages, kept_messages) =
            fold_fixture_with_kept(session.plan(), salt, kept_salt);
        let mask_coefficients = mask.coefficients().to_vec();
        let initial_main = main_messages.clone();
        let initial_mu = mask_messages.clone();
        session.fold_bind_row(row.clone()).unwrap();
        session.start_fold(&start.message).unwrap();
        session.fold_rounds_start().unwrap();
        let eps = session.epsilon().unwrap();
        let mut prover = FoldProverSession::new(
            &session.plan().fold,
            session.plan().fold_layout.as_ref().unwrap(),
            &mask,
            row.clone(),
            main_messages,
            mask_messages,
            kept_messages.clone(),
        )
        .unwrap();
        assert_eq!(prover.sum(), start.message.sum);
        prover.start(eps, &source).unwrap();
        let mut chain = session.chain.clone();
        let mut messages = Vec::new();
        let mut qs = Vec::new();
        for _ in 0..session.plan().fold.k {
            let q = prover.round().unwrap();
            qs.push(q);
            messages.push((q[0], q[2]));
            let mut message = chain.message();
            message.push_fp3(&q[0]);
            message.push_fp3(&q[2]);
            let prefix = message.prefix(KIND_ALGEBRAIC, 0).unwrap();
            chain.step(&prefix, 0).unwrap();
            let gamma = chain.draw_e().unwrap();
            session.receive_fold_round(q[0], q[2]).unwrap();
            assert_eq!(session.fold_gamma().last(), Some(&gamma));
            prover.bind(gamma).unwrap();
        }
        let output = session.fold_finish().unwrap();
        assert!(session.fold_next_stage().is_err());
        assert!(output.next_stage().is_err());
        let end = prover.end().unwrap();
        let main_end = prover.main_messages().to_vec();
        FoldRun {
            messages,
            qs,
            output,
            end,
            main_end,
            main_messages: initial_main,
            mask_messages: initial_mu,
            row,
            kept_messages,
            mask_coefficients,
            eps,
            start: start.message,
        }
    }

    fn fold_at(values: &[Fp3], at: &[Fp3], variables: u32) -> Vec<Fp3> {
        let mut out = values.to_vec();
        for challenge in at.iter().take(variables as usize) {
            out = out
                .chunks(2)
                .map(|pair| pair[0].add(&challenge.mul(&pair[1].sub(&pair[0]))))
                .collect();
        }
        out
    }

    fn fold_h_at(run: &FoldRun, x: &[Fp3]) -> Fp3 {
        let k = run.row.k;
        let h = run.row.h;
        let mut u = Fp3::ZERO;
        for variable in 0..k {
            let last = variable + 1 == k;
            let mut weight = Fp3::ZERO;
            for degree in 0..3u32 {
                if degree == 0 && !last {
                    continue;
                }
                let power = x[variable as usize].pow_u64(degree as u64);
                let index = if last {
                    2 * variable as usize + degree as usize
                } else {
                    2 * variable as usize + degree as usize - 1
                };
                weight = weight.add(&run.mask_coefficients[index].mul(&power));
            }
            for later in (variable + 1)..k {
                weight = weight.mul(&x[later as usize]);
            }
            u = u.add(&weight);
        }
        let main =
            fold_dot(&fold_at(&run.row.coefficients, x, k), &fold_at(&run.main_messages, x, k));
        let mut mask =
            fold_dot(&fold_at(&run.row.mask_coefficients, x, h), &fold_at(&run.mask_messages, x, h));
        for later in h..k {
            mask = mask.mul(&x[later as usize]);
        }
        let kept_value = run
            .row
            .kept
            .iter()
            .zip(&run.kept_messages)
            .fold(Fp3::ZERO, |sum, (form, message)| {
                sum.add(&fold_dot(&form.coefficients, message))
            });
        let mut kept = kept_value;
        for challenge in x.iter().take(k as usize) {
            kept = kept.mul(&Fp3::ONE.sub(challenge));
        }
        u.add(&run.eps.mul(&main.add(&mask).add(&kept)))
    }

    fn register_mask_start(session: &FoldeqSession, message: &FoldStartMessage) {
        let tree = *session
            .plan()
            .statement
            .mask_tree(session.plan().mask_tree_number)
            .unwrap();
        let statement = Arc::clone(&session.plan().statement);
        session
            .plan()
            .oracles
            .lock()
            .unwrap()
            .receive_mask(
                &statement,
                &tree,
                message.id,
                &message.notice,
                message.root,
            )
            .unwrap();
    }

    #[test]
    fn send_root_cannot_be_replaced_after_seeds_are_drawn() {
        let (mut session, _table) = open_gu_session();
        session.shared_seeds().unwrap();
        assert!(matches!(
            session.send_gu(&Digest([5; 32])),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn bat_chain_handoff_requires_the_read_stage() {
        let (session, _table) = open_gu_session();
        assert!(matches!(
            session.take_chain_for_bat(),
            Err(Error::InvalidSessionState)
        ));
    }

    #[test]
    fn wrong_prefix_chain_is_rejected() {
        let (plan, _table) = plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let wrong = Chain::begin(
            REQUEST_CHAIN_LABEL,
            &Digest::ZERO,
            b"another first query",
            &[],
            None,
        )
        .unwrap();
        assert!(FoldeqSession::open(plan, binder, wrong).is_err());
    }

    #[test]
    fn first_query_binds_the_plan_and_configuration() {
        let (plan, _table) = plan_fixture();
        assert!(FoldeqBinder::new(REQUEST_CHAIN_LABEL, Digest::ZERO, &plan).is_err());
        let mut other = plan.clone();
        other.column_draws = 1;
        assert_ne!(plan.first_query(), other.first_query());
        // The same root and record under another plan is not this first query.
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan.clone(), binder, chain).unwrap();
        assert!(session.bind_first_query(&other.first_query()).is_err());
        let own = plan.first_query();
        session.bind_first_query(&own).unwrap();
    }

    fn gamma_for(plan: FoldeqPlan) -> Vec<Fp3> {
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan, binder, chain).unwrap();
        let first = session.plan().first_query();
        session.bind_first_query(&first).unwrap();
        session.draw_gamma().unwrap();
        session.gamma().unwrap().to_vec()
    }

    #[test]
    fn binding_identity_and_canonical_fields_are_checked() {
        let table = committed();
        let reader = &table.reader;
        let parameters = binding_parameters(&reader.shape);
        let identity = reader_table_identity(reader).unwrap();
        let good = BindingStatement::new(
            PROOF_VERSION_WORD,
            parameters.protocol_identity,
            b"case".to_vec(),
            Digest([0x63; 32]),
            identity,
        )
        .unwrap();
        assert!(FoldeqBinding::new(&parameters, good.clone(), reader).is_ok());

        let mut changed_identity = good.clone();
        changed_identity.table_identity = Digest([0x64; 32]);
        assert!(FoldeqBinding::new(&parameters, changed_identity, reader).is_err());

        let mut changed_protocol = good.clone();
        changed_protocol.protocol_identity = Digest([0x65; 32]);
        assert!(FoldeqBinding::new(&parameters, changed_protocol, reader).is_err());

        assert!(BindingStatement::new(
            PROOF_VERSION_WORD,
            parameters.protocol_identity,
            Vec::<u8>::new(),
            Digest([0x63; 32]),
            identity,
        )
        .is_err());
        assert!(BindingStatement::new(
            PROOF_VERSION_WORD,
            parameters.protocol_identity,
            b"case".to_vec(),
            Digest::ZERO,
            identity,
        )
        .is_err());
        let huge = BindingStatement::new(
            PROOF_VERSION_WORD,
            parameters.protocol_identity,
            vec![b'a'; MAX_BINDING_STATEMENT_BYTES + 1],
            Digest([0x63; 32]),
            identity,
        )
        .unwrap();
        assert!(FoldeqBinding::new(&parameters, huge, reader).is_err());
    }

    #[test]
    fn five_binding_inputs_change_the_actual_session_prefix_or_first_challenge() {
        let (plan, _table) = plan_fixture();
        let parameters = binding_parameters(&plan.reader.shape);
        let base_first = plan.first_query();
        let base_start = *FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap()
        .start();
        let base_gamma = gamma_for(plan.clone());
        let request = Digest([0x63; 32]);

        let mut case_plan = plan.clone();
        case_plan.binding =
            binding_with(&case_plan.reader, &parameters, b"other-case", request);
        assert_ne!(base_first, case_plan.first_query());
        assert_ne!(
            base_gamma,
            gamma_for(case_plan.clone())
        );

        let mut request_plan = plan.clone();
        request_plan.binding =
            binding_with(&request_plan.reader, &parameters, b"case", Digest([0x66; 32]));
        assert_ne!(base_first, request_plan.first_query());
        assert_ne!(base_gamma, gamma_for(request_plan.clone()));

        let mut root_plan = plan.clone();
        root_plan.reader.root = Digest([0x67; 32]);
        root_plan.binding = binding_with(&root_plan.reader, &parameters, b"case", request);
        assert_ne!(base_first, root_plan.first_query());
        assert_ne!(base_gamma, gamma_for(root_plan.clone()));

        let mut record_plan = plan.clone();
        record_plan.reader.record.y[0] =
            record_plan.reader.record.y[0].add(&Fp3::ONE);
        record_plan.binding =
            binding_with(&record_plan.reader, &parameters, b"case", request);
        assert_ne!(base_first, record_plan.first_query());
        assert_ne!(base_gamma, gamma_for(record_plan.clone()));

        let mut config_plan = plan.clone();
        let changed_parameters = BindingParameters::new(
            PROOF_VERSION_WORD,
            parameters.protocol_identity,
            &RequestParams::default_with_code(
                plan.reader.shape.code.log_n,
                plan.reader.shape.code.ell,
                plan.reader.shape.code.r,
            )
            .unwrap(),
            Digest([0x68; 32]),
        )
        .unwrap();
        config_plan.reader.configuration = changed_parameters.digest().unwrap();
        config_plan.binding = binding_with(
            &config_plan.reader,
            &changed_parameters,
            b"case",
            request,
        );
        assert_ne!(base_first, config_plan.first_query());
        assert_ne!(base_gamma, gamma_for(config_plan.clone()));
        assert_ne!(
            base_start,
            *FoldeqBinder::new(
                REQUEST_CHAIN_LABEL,
                config_plan.reader.configuration,
                &config_plan,
            )
            .unwrap()
            .start()
        );
    }

    #[test]
    fn plan_and_binding_substitution_is_rejected_at_open_and_bind() {
        let (plan, _table) = plan_fixture();
        let parameters = binding_parameters(&plan.reader.shape);
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut changed = plan.clone();
        changed.binding =
            binding_with(&changed.reader, &parameters, b"other-case", Digest([0x63; 32]));
        assert!(FoldeqSession::open(changed, binder, chain).is_err());

        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan.clone(), binder, chain).unwrap();
        assert!(session.bind_first_query(&plan.first_query()).is_ok());

        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();
        let mut replaced_reader = plan.clone();
        replaced_reader.reader.root = Digest([0x69; 32]);
        assert!(FoldeqSession::open(replaced_reader, binder, chain).is_err());
        let mut other = plan.clone();
        other.binding =
            binding_with(&other.reader, &parameters, b"other-case", Digest([0x63; 32]));
        assert!(session.bind_first_query(&other.first_query()).is_err());
    }

    #[test]
    fn lin_gamma_is_the_psi_suffix() {
        let psi = vec![
            Fp3::from_u64(1),
            Fp3::from_u64(2),
            Fp3::from_u64(3),
            Fp3::from_u64(4),
        ];
        assert_eq!(
            lin_gamma(&psi, 2, 2).unwrap(),
            vec![Fp3::from_u64(3), Fp3::from_u64(4)]
        );
        assert!(lin_gamma(&psi, 3, 2).is_err());
        assert!(lin_gamma(&psi, 0, 5).is_err());
    }

    #[test]
    fn gamma_is_drawn_from_the_fixed_prefix() {
        let draw = |plan: FoldeqPlan| {
            let binder = FoldeqBinder::new(
                REQUEST_CHAIN_LABEL,
                plan.reader.configuration,
                &plan,
            )
            .unwrap();
            let chain = binder.chain().unwrap();
            let mut session = FoldeqSession::open(plan.clone(), binder, chain).unwrap();
            let first = plan.first_query();
            session.bind_first_query(&first).unwrap();
            session.draw_gamma().unwrap();
            session.gamma().unwrap().to_vec()
        };
        let (plan, _table) = plan_fixture();
        let first_query = plan.first_query();
        let first = draw(plan.clone());
        let (same, _table) = plan_fixture();
        assert_eq!(first, draw(same));
        let mut other = plan.clone();
        other.gamma_path = GammaPath::Direct {
            columns: other.reader.shape.v,
        };
        assert_ne!(first, draw(other.clone()));
        // A replaced LIN offset does not fit this table's plain columns and is
        // rejected by the plan fault before any draw; its first query is a
        // different binding.
        let mut offset = plan;
        offset.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding {
                matrix_rows: 1,
                k_lo: 1,
                k_col: 3,
            },
        };
        assert_ne!(first_query, offset.first_query());
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            offset.reader.configuration,
            &offset,
        );
        assert!(binder.is_err());
    }

    #[test]
    fn first_query_and_stage_order() {
        let (mut session, _table) = open_session();
        assert!(session.seeds().is_err());
        assert!(session.draw_gamma().is_err());
        let mut sent = session.plan().first_query();
        sent[0] ^= 1;
        assert!(session.bind_first_query(&sent).is_err());
        let good = session.plan().first_query();
        session.bind_first_query(&good).unwrap();
        assert_eq!(session.seeds().unwrap().len(), 1);
        assert!(session.send_gu(&Digest([4; 32])).is_err());
        assert!(session.draw_queries().is_err());
        assert!(session.read_queries(&[]).is_err());
        let plan = session.plan().clone();
        assert!(session.start_fold(&mask_start(&plan).message).is_err());
    }

    #[test]
    fn gamma_gu_queries_reads_and_fold_start() {
        let (mut session, table) = open_session();
        let plan = session.plan().clone();
        let good = plan.first_query();
        session.bind_first_query(&good).unwrap();
        let _seeds = session.seeds().unwrap();
        session.draw_gamma().unwrap();
        assert_eq!(session.gamma().unwrap().len(), plan.reader.shape.v as usize);
        let prover_hiding = HidingState::new(table.reader.root, 1, 8);
        let mut gu_prover =
            ProverFoldeq::new(prover_hiding, &table.reader, 1, plan.column_draws).unwrap();
        assert!(gu_prover.send_gu(&Digest([4; 32])).is_err());
        let charge = gu_prover.begin().unwrap();
        let gu = gu_prover.send_gu(&Digest([4; 32])).unwrap();
        session.send_gu(&gu.root).unwrap();
        assert_eq!(session.gu_root(), Some(&gu.root));
        assert_eq!(gu_prover.used(), plan.column_draws as u64);
        session.draw_queries().unwrap();
        assert_eq!(session.queries().len(), plan.column_draws);
        let mut prover = ProverReads::new(plan.ledger.lock().unwrap().max_reads());
        let openings = prover.open(&charge, &table, session.queries()).unwrap();
        let columns = session.read_queries(&openings).unwrap();
        assert_eq!(columns.len(), plan.column_draws);
        // A second read charges past the plan's budget.
        assert!(session.read_queries(&openings).is_err());
        let start = mask_start(&plan);
        let tree = plan.statement.mask_tree(plan.mask_tree_number).unwrap();
        session
            .plan()
            .oracles
            .lock()
            .unwrap()
            .receive_mask(
                &plan.statement,
                tree,
                9,
                &start.message.notice,
                start.message.root,
            )
            .unwrap();
        // Registration of a forged root is only a claim; the later opening
        // phase must reject it.
        let mut forged_registry = RuntimeOracles::default();
        forged_registry
            .receive_mask(
                &plan.statement,
                tree,
                9,
                &start.message.notice,
                Digest([5; 32]),
            )
            .unwrap();
        session.start_fold(&start.message).unwrap();
        assert!(session.epsilon().is_some());
        assert_eq!(session.start(), Some(&start.message));
        // The opening positions come from the verifier's own transcript; the
        // plan supplies only the count.
        session.draw_mask_queries().unwrap();
        let positions = session.mask_queries().to_vec();
        assert!(positions.len() <= plan.mask_query_count);
        let mask_openings = start.tree.open(&positions).unwrap();
        assert!(forged_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &positions,
                &mask_openings,
            )
            .is_err());
        let mut empty_registry = RuntimeOracles::default();
        empty_registry
            .receive_mask(
                &plan.statement,
                tree,
                9,
                &start.message.notice,
                start.message.root,
            )
            .unwrap();
        // An empty opening list is an explicit no-opening result, not a
        // successful mask authentication; it marks nothing, so a later
        // non-empty opening can still be verified. A list longer than the
        // plan's draw count is rejected.
        let no_op = empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &[],
                &RuntimeTreeOpening::default(),
            )
            .unwrap();
        assert_eq!(
            no_op,
            crate::pcs::request::fold::MaskAuthentication::NoOpening
        );
        let long = start.tree.open(&[0u64, 1, 2]).unwrap();
        assert!(empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &[0u64, 1, 2],
                &long,
            )
            .is_err());
        let verified = empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &positions,
                &mask_openings,
            )
            .unwrap();
        assert_eq!(
            verified,
            crate::pcs::request::fold::MaskAuthentication::Verified
        );
        assert!(empty_registry
            .authenticate_mask(
                tree,
                9,
                plan.mask_query_count,
                &positions,
                &mask_openings,
            )
            .is_err());
        session.read_mask(&mask_openings).unwrap();
        assert!(session.read_mask(&mask_openings).is_err());
        assert!(verify_mask_opening(
            tree,
            &Digest([5; 32]),
            &positions,
            &mask_openings
        )
        .is_err());
        let mut forged = mask_openings;
        forged.columns[0].symbols[0] = forged.columns[0].symbols[0] + Fp::ONE;
        assert!(verify_mask_opening(
            tree,
            &start.message.root,
            &positions,
            &forged
        )
        .is_err());
    }

    #[test]
    fn provider_hiding_is_bound_and_charged_before_send() {
        let (plan, table) = plan_fixture();
        let wrong_table = HidingState::new(Digest([9; 32]), 1, 8);
        assert!(ProverFoldeq::new(wrong_table, &table.reader, 1, plan.column_draws).is_err());
        let wrong_lifecycle = HidingState::new(table.reader.root, 2, 8);
        assert!(ProverFoldeq::new(wrong_lifecycle, &table.reader, 1, plan.column_draws).is_err());
        let too_small = HidingState::new(table.reader.root, 1, 1);
        let mut small =
            ProverFoldeq::new(too_small, &table.reader, 1, plan.column_draws).unwrap();
        assert!(small.begin().is_err());
        assert!(small.send_gu(&Digest([4; 32])).is_err());
        let state = HidingState::new(table.reader.root, 1, 8);
        let mut prover = ProverFoldeq::new(state, &table.reader, 1, plan.column_draws).unwrap();
        assert!(prover.send_gu(&Digest([4; 32])).is_err());
        let _charge = prover.begin().unwrap();
        assert!(prover.begin().is_err());
        assert_eq!(prover.used(), plan.column_draws as u64);
    }

    #[test]
    fn plan_faults_are_rejected() {
        let (plan, _table) = plan_fixture();
        let binder = FoldeqBinder::new(
            REQUEST_CHAIN_LABEL,
            plan.reader.configuration,
            &plan,
        )
        .unwrap();
        let chain = binder.chain().unwrap();

        let mut bad = plan.clone();
        bad.seed_budget = 0;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.column_draws = 0;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.mask_tree_number = 6;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.ledger = Arc::new(Mutex::new(ReadLedger::new(1)));
        bad.column_draws = 2;
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin { shape: LinShapeBinding { matrix_rows: 1, k_lo: 0, k_col: 1 } };
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // A high-bit matrix is rejected by the reference row-index bound
        // before any shift; the checked multiply stays as a defensive guard,
        // since within the row and shift bounds the product cannot overflow.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 1u64 << 63, k_lo: 1, k_col: 3 },
        };
        assert_eq!(bad.fault(), Some("the LIN shape binding has invalid numbers"));
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // A matrix that does not overflow but exceeds the plain columns is
        // refused by the upper-bound check.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 2, k_lo: 0, k_col: 2 },
        };
        assert_eq!(
            bad.fault(),
            Some("the matrix does not fit the table's plain columns")
        );
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        // The reference shape bound also limits the row index to 30 bits, so a
        // matrix with more than 2^30 rows is invalid regardless of the fit.
        let mut bad = plan.clone();
        bad.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: (1u64 << 30) + 1, k_lo: 0, k_col: 2 },
        };
        assert_eq!(bad.fault(), Some("the LIN shape binding has invalid numbers"));
        assert!(FoldeqSession::open(bad, binder.clone(), chain.clone()).is_err());

        let mut bad = plan;
        bad.group_id = 7;
        assert!(FoldeqSession::open(bad, binder, chain).is_err());
    }

    #[test]
    fn lin_row_bound_accepts_the_limit_without_materializing() {
        let (mut plan, _table) = plan_fixture();
        // A shape whose plain width admits the largest row count; only the
        // numbers are built, no row or matrix is allocated.
        let reader_shape = IndexShape {
            code: build_code_plan(32, (1u64 << 30) + 3, 1).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: LEAF_SALT_MIN_BYTES,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        plan.reader = IndexReader {
            shape: reader_shape,
            configuration: binding_parameters(&reader_shape).digest().unwrap(),
            root: Digest([7; 32]),
            record: IndexRecord { y: vec![Fp3::ZERO] },
        };
        plan.binding = binding_for(&plan.reader);
        plan.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: 1u64 << 30, k_lo: 0, k_col: 0 },
        };
        // The plain width is exactly 2^30, so this row count is legal.
        assert_eq!(plan.fault(), None);
        plan.gamma_path = GammaPath::Lin {
            shape: LinShapeBinding { matrix_rows: (1u64 << 30) + 1, k_lo: 0, k_col: 0 },
        };
        assert_eq!(plan.fault(), Some("the LIN shape binding has invalid numbers"));
    }

    // A local numbered source for the joint harness tests; it is not a
    // production randomness source and provides no hiding.
    struct TestCoins {
        seed: Digest,
        draw: u32,
    }

    impl TestCoins {
        fn new(seed: Digest) -> Self {
            Self { seed, draw: 0 }
        }

        fn value_at(&self, number: u32) -> Result<Fp3> {
            let mut coefficients = [Fp::ZERO; 3];
            for (coordinate, coefficient) in coefficients.iter_mut().enumerate() {
                *coefficient = crate::pcs::request::field::uniform_fp(|attempt| {
                    let digest = hash_labeled(
                        tag_word(TEST_COINS, 1 + coordinate as u64),
                        number as u64,
                        Some(&self.seed),
                        &attempt.to_le_bytes(),
                    )?;
                    Ok(u64::from_le_bytes(
                        digest.0[0..8].try_into().map_err(|_| Error::InvalidEncoding)?,
                    ))
                })?;
            }
            Ok(Fp3::new(coefficients[0], coefficients[1], coefficients[2]))
        }
    }

    impl crate::pcs::request::fold::MaskCoins for TestCoins {
        fn draw(&mut self) -> Result<(Fp3, u32)> {
            let number = self.draw;
            self.draw = self.draw.checked_add(1).ok_or(Error::ResourceLimit)?;
            Ok((self.value_at(number)?, number))
        }

        fn redraw(&self, number: u32) -> Result<Fp3> {
            self.value_at(number)
        }
    }

    #[test]
    fn gu_session_requires_seed_answers_record_then_queries() {
        let (mut session, _table) = open_gu_session();
        // The old skip path from the received root directly to draws is rejected.
        assert!(session.draw_queries().is_err());
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        session.shared_seeds().unwrap();
        assert_eq!(session.gu_shared_seeds().len(), 1);
        assert_eq!(
            session.gu_shared_seeds()[0].z.len(),
            gu_shape().sample_bits().unwrap() as usize
        );
        assert!(session.shared_seeds().is_err());
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(Fp3::from_u64(7)).unwrap();
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        assert!(session.draw_queries().is_err());
        session.join_gu_record().unwrap();
        assert!(session.join_gu_record().is_err());
        assert!(session.join_gu_columns(&[]).is_err());
        assert_eq!(session.pending_gu_rows().len(), 2);
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(300 + row + *position))
                        .collect(),
                )
            })
            .collect();
        let mut wrong = fixture.clone();
        if wrong.len() > 1 {
            wrong.pop();
        } else {
            let other = if wrong[0].0 == 0 { 1 } else { 0 };
            wrong.push((other, vec![Fp3::ZERO; session.plan().reader.shape.rows() as usize]));
        }
        assert!(session.join_gu_columns(&wrong).is_err());
        session.join_gu_columns(&fixture).unwrap();
        assert!(session.join_gu_columns(&fixture).is_err());
        assert!(session.draw_queries().is_err());
        let rows = session.pending_gu_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.iter().filter(|row| row.kind == GuRelationKind::Sample).count(), 1);
        assert_eq!(rows.iter().filter(|row| row.kind == GuRelationKind::Record).count(), 1);
        assert_eq!(session.pending_column_count(), queries.len());
        for draw in 0..session.pending_column_count() {
            let column = session.column_relation_row(draw).unwrap();
            assert_eq!(column.oracle, 10);
            assert_eq!(column.group, 1);
            assert_eq!(column.kind, GuRelationKind::Column);
        }
    }

    #[test]
    fn gu_session_rejects_missing_and_out_of_order_stages() {
        let (plan, _table) = gu_plan_fixture();
        let binder = FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).unwrap();
        let chain = binder.chain().unwrap();
        let mut session = FoldeqSession::open(plan, binder, chain).unwrap();
        assert!(session.shared_seeds().is_err());
        session
            .bind_first_query(&session.plan().first_query())
            .unwrap();
        assert!(session.shared_seeds().is_err());
        session.seeds().unwrap();
        session.draw_gamma().unwrap();
        assert!(session.draw_queries().is_err());
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        assert!(session.shared_seeds().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_columns(&[]).is_err());
        session.join_gu_record().unwrap();
        assert!(session.join_gu_columns(&[]).is_err());
        assert!(session.read_queries(&[]).is_err());
    }

    #[test]
    fn gu_shared_seeds_are_drawn_from_the_received_root() {
        let (mut first, _) = open_gu_with_root(Digest([4; 32]));
        let (mut second, _) = open_gu_with_root(Digest([4; 32]));
        first.shared_seeds().unwrap();
        second.shared_seeds().unwrap();
        assert_eq!(first.transcript_value(), second.transcript_value());
        assert_eq!(first.gu_shared_seeds(), second.gu_shared_seeds());

        let (mut changed, _) = open_gu_with_root(Digest([5; 32]));
        assert_ne!(first.transcript_value(), changed.transcript_value());
        changed.shared_seeds().unwrap();
        assert_ne!(first.gu_shared_seeds(), changed.gu_shared_seeds());
    }

    #[test]
    fn gu_answer_round_feeds_the_following_draws() {
        let (mut first, _) = open_gu_with_root(Digest([4; 32]));
        first.shared_seeds().unwrap();
        first.receive_gu_answer(Fp3::ONE).unwrap();
        first.join_gu_record().unwrap();
        let first_value = *first.transcript_value();
        first.draw_queries().unwrap();

        let (mut second, _) = open_gu_with_root(Digest([4; 32]));
        second.shared_seeds().unwrap();
        second.receive_gu_answer(Fp3::from_u64(2)).unwrap();
        second.join_gu_record().unwrap();
        let second_value = *second.transcript_value();
        second.draw_queries().unwrap();

        assert_ne!(first_value, second_value);
        assert_ne!(first.queries(), second.queries());
    }

    #[test]
    fn gu_session_joins_the_trusted_group_once() {
        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        assert!(session
            .pending_gu_rows()
            .iter()
            .all(|row| row.oracle == 10 && row.group == 1));

        let (wrong_code, _) = gu_plan_with(
            Some(build_code_plan(3, 8, 0).unwrap()),
            vec![1, 2],
            10,
        );
        let mut wrong = open_gu_plan_from(wrong_code);
        wrong.send_gu(&Digest([4; 32])).unwrap();
        wrong.shared_seeds().unwrap();
        wrong.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(wrong.join_gu_record().is_err());

        let (duplicate, _) = gu_plan_with(
            Some(build_code_plan(3, 6, 1).unwrap()),
            vec![10],
            10,
        );
        let mut duplicate = open_gu_plan_from(duplicate);
        duplicate.send_gu(&Digest([4; 32])).unwrap();
        duplicate.shared_seeds().unwrap();
        duplicate.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(duplicate.join_gu_record().is_err());
    }

    #[test]
    fn gu_session_requires_joined_columns_before_fold_and_uses_verified_reads() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        let mut tampered = ordered.clone();
        tampered[0].symbols[0] = tampered[0].symbols[0] + Fp::ONE;
        assert!(session.read_queries(&tampered).is_err());
        let columns = session.read_queries(&ordered).unwrap();
        assert_eq!(columns.len(), queries.len());
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(700 + row + *position))
                        .collect(),
                )
            })
            .collect();
        assert!(session.join_gu_columns(&fixture).is_err());
        let start = mask_start(session.plan());
        let tree = *session
            .plan()
            .statement
            .mask_tree(session.plan().mask_tree_number)
            .unwrap();
        let statement = Arc::clone(&session.plan().statement);
        session
            .plan()
            .oracles
            .lock()
            .unwrap()
            .receive_mask(
                &statement,
                &tree,
                start.message.id,
                &start.message.notice,
                start.message.root,
            )
            .unwrap();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        session.fold_bind_row(row).unwrap();
        assert!(session.start_fold(&start.message).is_err());
        session.join_gu_read_columns().unwrap();
        assert!(session.join_gu_read_columns().is_err());
        session.start_fold(&start.message).unwrap();
        assert_eq!(session.pending_gu_rows().len(), 2);
        assert_eq!(session.pending_column_count(), queries.len());
    }

    #[test]
    fn gu_session_rows_check_one_independent_witness() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        let gu = session.plan().gu.clone().unwrap();
        let shape = session.plan().reader.shape;
        let eq_gamma = foldeq_weights(session.gamma().unwrap(), shape.v).unwrap();
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
        let mut folded_message = vec![Fp3::ZERO; shape.code.ell as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in messages[row].iter().enumerate() {
                folded_message[entry] =
                    folded_message[entry].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut folded_tail = vec![Fp3::ZERO; shape.code.r as usize];
        for (row, weight) in eq_gamma.iter().enumerate() {
            for (entry, value) in tails[row].iter().enumerate() {
                folded_tail[entry] =
                    folded_tail[entry].add(&weight.mul(&Fp3::from_base(*value)));
            }
        }
        let mut witness = gu_table(&gu, &folded_message, &folded_tail).unwrap();
        let row_bits = 1u64;
        let pad_virtual = (gu.u.main_rows - 1)
            + ((gu.u.code.ell - gu.u.answers as u64) << row_bits);
        let pad = ((pad_virtual & ((1u64 << row_bits) - 1)) * gu.u.code.ell
            + (pad_virtual >> row_bits)) as usize;
        witness.main[pad] = Fp3::from_u64(999);

        let notice = GuNotice {
            root: *session.gu_root().unwrap(),
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        };
        let mut probe_registry = GuRegistry::new();
        probe_registry.receive(&gu, &notice).unwrap();
        let seeds = session.gu_shared_seeds().to_vec();
        let sample_probe =
            sample_row(&gu, &probe_registry, &seeds, 0, Fp3::ZERO).unwrap();
        let sample_target = sample_probe.evaluate(&witness).unwrap();
        session.receive_gu_answer(sample_target).unwrap();
        session.join_gu_record().unwrap();
        let rows = session.pending_gu_rows();
        let sample = rows.iter().find(|row| row.kind == GuRelationKind::Sample).unwrap();
        let record = rows.iter().find(|row| row.kind == GuRelationKind::Record).unwrap();
        assert_eq!(sample.evaluate(&witness).unwrap(), sample.target);
        assert_eq!(record.evaluate(&witness).unwrap(), record.target);

        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        session.join_gu_read_columns().unwrap();
        assert_eq!(session.pending_column_count(), queries.len());
        for draw in 0..session.pending_column_count() {
            let column = session.column_relation_row(draw).unwrap();
            assert_eq!(column.evaluate(&witness).unwrap(), column.target);
        }
    }

    #[test]
    fn gu_session_reuses_one_verified_column_for_repeated_draws() {
        let mut found = None;
        for answer in 0..64u64 {
            let (mut plan, table) = gu_plan_fixture();
            plan.column_draws = 8;
            plan.ledger = Arc::new(Mutex::new(ReadLedger::new(8)));
            let mut session = open_gu_plan_from(plan);
            session.send_gu(&Digest([4; 32])).unwrap();
            session.shared_seeds().unwrap();
            session
                .receive_gu_answer(Fp3::from_u64(answer))
                .unwrap();
            session.join_gu_record().unwrap();
            session.draw_queries().unwrap();
            if unique_sorted(session.queries()).len() < session.queries().len() {
                found = Some((session, table));
                break;
            }
        }
        let (mut session, table) = found.expect("a repeated draw in the searched range");
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        let mut tampered = ordered.clone();
        let mut seen = std::collections::HashSet::new();
        let mut changed = false;
        for (index, position) in queries.iter().enumerate() {
            if !seen.insert(*position) {
                tampered[index].symbols[0] = tampered[index].symbols[0] + Fp::ONE;
                changed = true;
                break;
            }
        }
        assert!(changed);
        assert!(session.read_queries(&tampered).is_err());
        session.read_queries(&ordered).unwrap();
        assert_eq!(
            session.plan().ledger.lock().unwrap().used(),
            unique.len() as u64
        );
        assert_eq!(session.plan().column_draws, queries.len());
        session.join_gu_read_columns().unwrap();
        assert_eq!(session.pending_column_count(), queries.len());
        let mut targets = std::collections::HashMap::new();
        let mut repeats = 0usize;
        for (draw, position) in queries.iter().enumerate() {
            let row = session.column_relation_row(draw).unwrap();
            match targets.get(position) {
                Some(target) => {
                    assert_eq!(*target, row.target);
                    repeats += 1;
                }
                None => {
                    targets.insert(*position, row.target);
                }
            }
        }
        assert!(repeats > 0);
    }

    #[test]
    fn gu_session_rejects_a_rebound_oracle_tree_or_group() {
        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().oracle = 11;
        assert!(session.join_gu_record().is_err());

        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().tree.tree_number += 1;
        assert!(session.join_gu_record().is_err());

        let (mut session, _) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.plan.gu.as_mut().unwrap().group = 2;
        assert!(session.join_gu_record().is_err());
    }

    #[test]
    fn gu_session_closes_every_ordered_answer() {
        let (plan, _table) = gu_plan_with_answers(2);
        let mut session = open_gu_plan_from(plan);
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        assert_eq!(session.gu_shared_seeds().len(), 2);
        let before = *session.transcript_value();
        let gu = session.plan().gu.clone().unwrap();
        let notice = GuNotice {
            root: *session.gu_root().unwrap(),
            oracle: gu.oracle,
            tree_number: gu.tree.tree_number,
            part: gu.part,
        };
        let mut probe = GuRegistry::new();
        probe.receive(&gu, &notice).unwrap();
        let seeds = session.gu_shared_seeds().to_vec();
        let witness = GuWitness {
            main: vec![
                Fp3::ONE;
                usize::try_from(gu.u.main_rows * gu.u.code.ell).unwrap()
            ],
            mask: Vec::new(),
        };
        let first = sample_row(&gu, &probe, &seeds, 0, Fp3::ZERO)
            .unwrap()
            .evaluate(&witness)
            .unwrap();
        let second = sample_row(&gu, &probe, &seeds, 1, Fp3::ZERO)
            .unwrap()
            .evaluate(&witness)
            .unwrap();
        assert_ne!(first, second);

        session.receive_gu_answer(first).unwrap();
        assert_eq!(*session.transcript_value(), before);
        assert!(session.draw_queries().is_err());
        assert!(session.join_gu_record().is_err());
        session.receive_gu_answer(second).unwrap();
        let after = *session.transcript_value();
        assert_ne!(after, before);
        assert!(session.receive_gu_answer(Fp3::ONE).is_err());
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let samples: Vec<_> = session
            .pending_gu_rows()
            .iter()
            .filter(|row| row.kind == GuRelationKind::Sample)
            .collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].target, first);
        assert_eq!(samples[1].target, second);

        let run = |first: Fp3, second: Fp3| {
            let (plan, _table) = gu_plan_with_answers(2);
            let mut session = open_gu_plan_from(plan);
            session.send_gu(&Digest([4; 32])).unwrap();
            session.shared_seeds().unwrap();
            let before = *session.transcript_value();
            session.receive_gu_answer(first).unwrap();
            let middle = *session.transcript_value();
            session.receive_gu_answer(second).unwrap();
            let after = *session.transcript_value();
            (before, middle, after)
        };
        let baseline = run(first, second);
        let first_changed = run(first.add(&Fp3::ONE), second);
        let second_changed = run(first, second.add(&Fp3::ONE));
        assert_eq!(baseline.0, baseline.1);
        assert_ne!(baseline.2, baseline.0);
        assert_ne!(baseline.2, first_changed.2);
        assert_ne!(baseline.2, second_changed.2);
    }

    #[test]
    fn gu_fixture_columns_do_not_pass_fold() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let fixture: Vec<(u64, Vec<Fp3>)> = unique
            .iter()
            .map(|position| {
                (
                    *position,
                    (0..session.plan().reader.shape.rows())
                        .map(|row| Fp3::from_u64(900 + row + *position))
                        .collect(),
                )
            })
            .collect();
        session.join_gu_columns(&fixture).unwrap();
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        assert!(session.join_gu_read_columns().is_err());
        let start = mask_start(session.plan());
        register_mask_start(&session, &start.message);
        assert!(session.start_fold(&start.message).is_err());
    }

    #[test]
    fn gu_wrong_fold_group_is_rejected() {
        let (plan, _table) = gu_plan_two_groups(2);
        assert!(plan.fault().is_some());
        assert!(FoldeqBinder::new(REQUEST_CHAIN_LABEL, plan.reader.configuration, &plan).is_err());
    }

    #[test]
    fn gu_fresh_id_collision_is_rejected() {
        let (plan, _table) = gu_plan_fresh_collision();
        let mut session = open_gu_plan_from(plan);
        session.send_gu(&Digest([4; 32])).unwrap();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        assert!(session.join_gu_record().is_err());
    }

    #[test]
    fn gu_fold_consumes_the_joined_membership() {
        let (mut session, table) = open_gu_session();
        session.shared_seeds().unwrap();
        session.receive_gu_answer(Fp3::ONE).unwrap();
        session.join_gu_record().unwrap();
        session.draw_queries().unwrap();
        let queries = session.queries().to_vec();
        let unique = unique_sorted(&queries);
        let openings = table.open_columns(&unique).unwrap();
        let mut by_position = std::collections::HashMap::new();
        for opening in &openings {
            by_position.insert(opening.position, opening.clone());
        }
        let ordered: Vec<ColumnOpening> = queries
            .iter()
            .map(|position| by_position[position].clone())
            .collect();
        session.read_queries(&ordered).unwrap();
        session.join_gu_read_columns().unwrap();
        let start = mask_start(session.plan());
        register_mask_start(&session, &start.message);
        let membership = session.gu_membership.as_mut().unwrap();
        membership.members.push(9);
        membership.members.sort_unstable();
        assert!(session.start_fold(&start.message).is_err());
    }

    fn fold_poly_at(q: &[Fp3; 3], at: Fp3) -> Fp3 {
        q[0]
            .add(&q[1].mul(&at))
            .add(&q[2].mul(&at.mul(&at)))
    }

    fn fold_low_manual(values: Vec<Fp3>, gamma: &[Fp3], variables: u32) -> Vec<Fp3> {
        let mut out = values;
        for challenge in gamma.iter().take(variables as usize) {
            out = out
                .chunks(2)
                .map(|pair| pair[0].add(&challenge.mul(&pair[1].sub(&pair[0]))))
                .collect();
        }
        out
    }

    fn fold_q_direct(run: &FoldRun, round: usize, at: Fp3) -> Fp3 {
        let k = run.row.k as usize;
        let later = k - round - 1;
        let mut sum = Fp3::ZERO;
        for bits in 0..(1usize << later) {
            let mut x = Vec::with_capacity(k);
            x.extend_from_slice(&run.output.gamma[..round]);
            x.push(at);
            for variable in 0..later {
                x.push(Fp3::from_u64(((bits >> variable) & 1) as u64));
            }
            sum = sum.add(&fold_h_at(run, &x));
        }
        sum
    }

    fn fold_kept_value(run: &FoldRun) -> Fp3 {
        run.row
            .kept
            .iter()
            .zip(&run.kept_messages)
            .fold(Fp3::ZERO, |sum, (form, message)| {
                sum.add(&fold_dot(&form.coefficients, message))
            })
    }

    fn assert_fold_run_direct(run: &FoldRun) {
        for round in 0..run.row.k as usize {
            for degree in 0..3u64 {
                let at = Fp3::from_u64(degree);
                assert_eq!(
                    fold_poly_at(&run.qs[round], at),
                    fold_q_direct(run, round, at)
                );
            }
        }
        let kept_value = fold_kept_value(run);
        let end_value = fold_dot(&run.output.coefficients, &run.main_end)
            .add(&fold_dot(&run.output.mask_coefficients, &run.end.mu))
            .add(&fold_dot(&run.output.fresh_coefficients, &run.end.fresh))
            .add(&run.output.kept_factor.mul(&kept_value));
        assert_eq!(end_value, run.output.target);
        assert_eq!(run.output.kept.len(), run.row.kept.len());
        for (out, source) in run.output.kept.iter().zip(&run.row.kept) {
            assert_eq!(out.oracle, source.oracle);
            assert_eq!(
                out.coefficients,
                source
                    .coefficients
                    .iter()
                    .map(|value| value.mul(&run.output.kept_factor))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            run.output.kept_rows,
            run.output.layout.kept_rows_checked().unwrap()
        );
    }

    #[test]
    fn fold_session_rounds_produce_the_final_constraint() {
        let run = run_fold_instance(1);
        assert_eq!(run.row.k, 2);
        assert_eq!(run.messages.len(), run.row.k as usize);
        assert_ne!(run.messages[0], run.messages[1]);
        assert_eq!(run.output.group, run.row.group);
        assert_eq!(run.output.members, run.row.members);
        assert_eq!(run.output.fresh, run.start.id);
        assert_eq!(run.output.fresh_root, run.start.root);
        assert_eq!(run.output.tree_number, run.start.tree_number);
        assert_eq!(run.output.gamma.len(), run.row.k as usize);
        assert_eq!(run.output.sum, run.start.sum);
        assert_eq!(run.output.eps, run.eps);

        let mut expected_c = fold_low_manual(run.row.coefficients.clone(), &run.output.gamma, run.row.k);
        for value in &mut expected_c {
            *value = value.mul(&run.output.eps);
        }
        assert_eq!(run.output.coefficients, expected_c);
        let mut expected_d =
            fold_low_manual(run.row.mask_coefficients.clone(), &run.output.gamma, run.row.h);
        let mut scale = run.output.eps;
        for challenge in run.output.gamma.iter().skip(run.row.h as usize) {
            scale = scale.mul(challenge);
        }
        for value in &mut expected_d {
            *value = value.mul(&scale);
        }
        assert_eq!(run.output.mask_coefficients, expected_d);

        let kept_value = run
            .row
            .kept
            .iter()
            .zip(&run.kept_messages)
            .fold(Fp3::ZERO, |sum, (form, message)| {
                sum.add(&fold_dot(&form.coefficients, message))
            });
        let end_value = fold_dot(&run.output.coefficients, &run.main_end)
            .add(&fold_dot(&run.output.mask_coefficients, &run.end.mu))
            .add(&fold_dot(&run.output.fresh_coefficients, &run.end.fresh))
            .add(&run.output.kept_factor.mul(&kept_value));
        assert_eq!(end_value, run.output.target);
        assert_eq!(run.output.kept.len(), run.row.kept.len());
        for (out, source) in run.output.kept.iter().zip(&run.row.kept) {
            assert_eq!(out.oracle, source.oracle);
            assert_eq!(
                out.coefficients,
                source
                    .coefficients
                    .iter()
                    .map(|value| value.mul(&run.output.kept_factor))
                    .collect::<Vec<_>>()
            );
        }
        assert!(run.output.next_stage().is_err());

        for round in 0..run.row.k as usize {
            for degree in 0..3u64 {
                let at = Fp3::from_u64(degree);
                assert_eq!(
                    fold_poly_at(&run.qs[round], at),
                    fold_q_direct(&run, round, at)
                );
            }
        }
    }

    #[test]
    fn fold_session_fixed_input_change_changes_rounds() {
        let first = run_fold_instance(1);
        let second = run_fold_instance(2);
        assert_ne!(first.messages, second.messages);
        assert_ne!(first.output.target, second.output.target);
    }

    #[test]
    fn fold_session_kept_layout_is_enforced() {
        let (mut session, _table, _start, _mask, _source) = fold_open_prestart();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        let ell = session.plan().fold.code.ell as usize;
        let form = |oracle: u64, rows: usize| FoldKeptForm {
            oracle,
            coefficients: (0..rows * ell)
                .map(|index| Fp3::from_u64(2001 + index as u64))
                .collect(),
        };
        let mut ordinary = row.clone();
        ordinary.kept = vec![form(10, 2)];
        assert!(session.fold_bind_row(ordinary).is_err());
        let mut wrong_rows = row.clone();
        wrong_rows.kept = vec![form(2, 1)];
        assert!(session.fold_bind_row(wrong_rows).is_err());
        let mut mixed = row.clone();
        mixed.kept = vec![form(2, 2), form(10, 2)];
        assert!(session.fold_bind_row(mixed).is_err());
        let mut duplicate = row.clone();
        duplicate.kept = vec![form(2, 2), form(2, 2)];
        assert!(session.fold_bind_row(duplicate).is_err());
        session.fold_bind_row(row.clone()).unwrap();

        let (mut single, _table, start, _mask, _source) = fold_open_prestart();
        let (mut single_row, _, _, _) = fold_fixture(single.plan(), 1);
        single_row.kept = vec![form(1, 1)];
        single.fold_bind_row(single_row).unwrap();
        single.start_fold(&start.message).unwrap();
        single.fold_rounds_start().unwrap();
    }

    #[test]
    fn fold_session_kept_evidence_is_direct_and_nonzero() {
        let baseline = run_fold_instance_kept(1, 1);
        let changed = run_fold_instance_kept(1, 2);
        assert_eq!(baseline.main_messages, changed.main_messages);
        assert_eq!(baseline.mask_messages, changed.mask_messages);
        assert_eq!(baseline.mask_coefficients, changed.mask_coefficients);
        assert_ne!(baseline.kept_messages, changed.kept_messages);
        let first_kept = fold_kept_value(&baseline);
        let second_kept = fold_kept_value(&changed);
        assert_ne!(first_kept, Fp3::ZERO);
        assert_ne!(second_kept, Fp3::ZERO);
        assert_ne!(first_kept, second_kept);
        assert_ne!(baseline.eps, Fp3::ZERO);
        assert_ne!(baseline.output.kept_factor, Fp3::ZERO);
        assert_ne!(changed.output.kept_factor, Fp3::ZERO);
        assert_ne!(baseline.qs, changed.qs);
        assert_ne!(baseline.output.target, changed.output.target);
        assert_fold_run_direct(&baseline);
        assert_fold_run_direct(&changed);
    }

    #[test]
    fn fold_layout_mask_only_and_row_overflow_are_checked() {
        let (plan, _table) = gu_plan_fixture();
        let layout = plan.fold_layout.clone().unwrap();
        assert!(layout.fault(&plan.fold).is_none());

        let mut mask_only = layout.clone();
        mask_only.members.push(FoldMemberLayout {
            oracle: 11,
            main_rows: 0,
            mask_rows: 1,
            kept: false,
        });
        assert!(mask_only.fault(&plan.fold).is_none());
        let (mut mask_row, _, _, _) = fold_fixture(&plan, 1);
        mask_row.members = mask_only
            .members
            .iter()
            .map(|member| member.oracle)
            .collect();
        assert!(mask_row.check(&plan.fold, &mask_only).is_ok());
        let mut kept_mask = mask_only.clone();
        kept_mask.members.last_mut().unwrap().kept = true;
        assert!(kept_mask.fault(&plan.fold).is_some());

        let (mut session, _table, _start, _mask, _source) = fold_open_prestart();
        session.plan.fold_layout = Some(mask_only.clone());
        let (mut outside_member, _, _, _) = fold_fixture(session.plan(), 1);
        outside_member.members = mask_only
            .members
            .iter()
            .map(|member| member.oracle)
            .collect();
        assert!(session.fold_bind_row(outside_member).is_err());
        let mut empty = mask_only.clone();
        empty.members.last_mut().unwrap().mask_rows = 0;
        assert!(empty.fault(&plan.fold).is_some());

        let mut main_overflow = layout.clone();
        main_overflow.members[0].kept = false;
        main_overflow.members.push(FoldMemberLayout {
            oracle: 11,
            main_rows: 1,
            mask_rows: 0,
            kept: false,
        });
        assert!(main_overflow.fault(&plan.fold).is_some());

        let mut mask_overflow = layout.clone();
        for (offset, oracle) in [11u64, 12, 13].into_iter().enumerate() {
            mask_overflow.members.push(FoldMemberLayout {
                oracle,
                main_rows: 0,
                mask_rows: 1,
                kept: false,
            });
            if offset < 2 {
                assert!(mask_overflow.fault(&plan.fold).is_none());
            } else {
                assert!(mask_overflow.fault(&plan.fold).is_some());
            }
        }

        let mut kept_overflow = layout.clone();
        kept_overflow.members = vec![
            FoldMemberLayout {
                oracle: 1,
                main_rows: u64::MAX,
                mask_rows: 0,
                kept: true,
            },
            FoldMemberLayout {
                oracle: 2,
                main_rows: 2,
                mask_rows: 0,
                kept: true,
            },
        ];
        assert!(kept_overflow.fault(&plan.fold).is_some());
        let mut plan_overflow = plan.clone();
        plan_overflow.fold_layout = Some(kept_overflow);
        assert!(plan_overflow.fault().is_some());
    }

    #[test]
    fn fold_session_kept_rows_three_virtual_mapping_is_independent() {
        let run = run_fold_instance_kept_rows(1, 1, Some(3));
        let ell = run.output.code.ell as usize;
        assert!(ell > 1);
        let rows = 3usize;
        let row_bits = 2usize;
        let virtual_rows = 1usize << row_bits;
        let coefficients = &run.row.kept[0].coefficients;
        let messages = &run.kept_messages[0];
        assert_eq!(coefficients.len(), rows * ell);
        assert_eq!(messages.len(), rows * ell);

        let mut virtual_coefficients = vec![Fp3::ZERO; virtual_rows * ell];
        let mut virtual_messages = vec![Fp3::ZERO; virtual_rows * ell];
        for row in 0..rows {
            for place in 0..ell {
                let compact = row * ell + place;
                let virtual_index = row + virtual_rows * place;
                virtual_coefficients[virtual_index] = coefficients[compact];
                virtual_messages[virtual_index] = messages[compact];
            }
        }
        for place in 0..ell {
            assert_eq!(virtual_coefficients[rows + virtual_rows * place], Fp3::ZERO);
            assert_eq!(virtual_messages[rows + virtual_rows * place], Fp3::ZERO);
        }
        let manual_dot = |left: &[Fp3], right: &[Fp3]| {
            left.iter()
                .zip(right)
                .fold(Fp3::ZERO, |sum, (a, b)| sum.add(&a.mul(b)))
        };
        let compact_dot = manual_dot(coefficients, messages);
        let virtual_dot = manual_dot(&virtual_coefficients, &virtual_messages);
        assert_ne!(compact_dot, Fp3::ZERO);
        assert_eq!(compact_dot, virtual_dot);
        assert_eq!(virtual_coefficients[2 + virtual_rows * (ell - 1)], coefficients[2 * ell + ell - 1]);

        let factor = run.output.kept_factor;
        assert_eq!(
            run.output.kept[0].coefficients,
            coefficients
                .iter()
                .map(|value| value.mul(&factor))
                .collect::<Vec<_>>()
        );
        let mut virtual_output = vec![Fp3::ZERO; virtual_rows * ell];
        for row in 0..rows {
            for place in 0..ell {
                virtual_output[row + virtual_rows * place] =
                    run.output.kept[0].coefficients[row * ell + place];
            }
        }
        for place in 0..ell {
            assert_eq!(virtual_output[rows + virtual_rows * place], Fp3::ZERO);
        }
        assert_eq!(manual_dot(&virtual_output, &virtual_messages), factor.mul(&compact_dot));

        let mut swapped = vec![Fp3::ZERO; virtual_rows * ell];
        for row in 0..rows {
            for place in 0..ell {
                swapped[place + virtual_rows * row] = coefficients[row * ell + place];
            }
        }
        assert_ne!(swapped, virtual_coefficients);
        assert_ne!(manual_dot(&swapped, &virtual_messages), virtual_dot);

        let (mut session, _table, _start, _mask, _source) = fold_open_prestart();
        let layout = session.plan.fold_layout.as_mut().unwrap();
        for member in &mut layout.members {
            if member.oracle == 2 {
                member.main_rows = 3;
            }
        }
        assert!(session.plan.fault().is_none());
        let (mut padded, _, _, _) = fold_fixture(session.plan(), 1);
        padded.kept[0].coefficients.extend(vec![Fp3::ZERO; ell]);
        assert!(session.fold_bind_row(padded).is_err());
        assert_fold_run_direct(&run);
    }

    #[test]
    fn fold_session_kept_rows_three_use_compact_mapping() {
        let run = run_fold_instance_kept_rows(1, 1, Some(3));
        assert_eq!(run.output.layout.member(2).unwrap().main_rows, 3);
        assert_eq!(run.row.kept[0].coefficients.len(), 3 * 4);
        assert_eq!(run.output.kept[0].coefficients.len(), 3 * 4);
        assert!(run.output.kept_rows >= 3);
        assert_fold_run_direct(&run);

        let (mut session, _table, _start, _mask, _source) = fold_open_prestart();
        let layout = session.plan.fold_layout.as_mut().unwrap();
        for member in &mut layout.members {
            if member.oracle == 2 {
                member.main_rows = 3;
            }
        }
        assert!(session.plan.fault().is_none());
        let (mut row, _, _, _) = fold_fixture(session.plan(), 1);
        row.kept[0].coefficients.push(Fp3::ONE);
        assert!(session.fold_bind_row(row).is_err());
    }

    #[test]
    fn fold_session_round_order_and_missing_rounds_fail_closed() {
        let (mut session, _table, _start, _mask, _source) = fold_open_ready();
        assert!(session.draw_mask_queries().is_err());
        assert!(session.fold_finish().is_err());
        assert!(session.receive_fold_round(Fp3::ONE, Fp3::ONE).is_err());
        session.fold_rounds_start().unwrap();
        assert!(session.fold_finish().is_err());
        session.receive_fold_round(Fp3::ONE, Fp3::from_u64(2)).unwrap();
        assert!(session.fold_finish().is_err());
        session
            .receive_fold_round(Fp3::from_u64(3), Fp3::from_u64(4))
            .unwrap();
        assert!(session
            .receive_fold_round(Fp3::from_u64(5), Fp3::from_u64(6))
            .is_err());
        session.fold_finish().unwrap();
        assert!(session.fold_finish().is_err());
        assert!(session.fold_next_stage().is_err());
    }

    #[test]
    fn fold_session_fixes_input_before_epsilon() {
        let (mut session, _table, start, _mask, _source) = fold_open_prestart();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        session.fold_bind_row(row.clone()).unwrap();
        assert!(session.fold_bind_row(row.clone()).is_err());
        session.start_fold(&start.message).unwrap();
        let mut changed_coefficients = row.clone();
        changed_coefficients.coefficients[0] =
            changed_coefficients.coefficients[0].add(&Fp3::ONE);
        assert!(session.fold_bind_row(changed_coefficients).is_err());
        let mut changed_target = row.clone();
        changed_target.target = changed_target.target.add(&Fp3::ONE);
        assert!(session.fold_bind_row(changed_target).is_err());
        session.fold_rounds_start().unwrap();
    }

    #[test]
    fn fold_session_wrong_group_layout_or_shape_is_rejected() {
        let (mut session, _table, _start, _mask, _source) = fold_open_prestart();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        let mut wrong_group = row.clone();
        wrong_group.group += 1;
        assert!(session.fold_bind_row(wrong_group).is_err());
        let mut wrong_rows = row.clone();
        wrong_rows.k = 1;
        assert!(session.fold_bind_row(wrong_rows).is_err());
        let mut wrong_members = row.clone();
        wrong_members.members.pop();
        assert!(session.fold_bind_row(wrong_members).is_err());
        let mut wrong_shape = row.clone();
        wrong_shape.coefficients.pop();
        assert!(session.fold_bind_row(wrong_shape).is_err());
        session.fold_bind_row(row.clone()).unwrap();
        assert!(session.fold_bind_row(row.clone()).is_err());
        let start = _start;
        session.start_fold(&start.message).unwrap();
        assert!(session.fold_bind_row(row.clone()).is_err());
        let mut changed_coefficients = row.clone();
        changed_coefficients.coefficients[0] =
            changed_coefficients.coefficients[0].add(&Fp3::ONE);
        assert!(session.fold_bind_row(changed_coefficients).is_err());
        let mut changed_target = row.clone();
        changed_target.target = changed_target.target.add(&Fp3::ONE);
        assert!(session.fold_bind_row(changed_target).is_err());
        session.fold_rounds_start().unwrap();
    }

    #[test]
    fn fold_session_wrong_root_is_rejected_before_rounds() {
        let (mut session, _table, start, _mask, _source) = fold_open_prestart();
        let (row, _, _, _) = fold_fixture(session.plan(), 1);
        session.fold_bind_row(row).unwrap();
        let mut bad = start.message.clone();
        bad.root = Digest([9; 32]);
        assert!(session.start_fold(&bad).is_err());
        assert!(session.fold_rounds_start().is_err());
    }

    #[test]
    fn fold_prover_second_as_drawn_check_rejects_another_source() {
        let (session, _table, start, mask, _source) = fold_open_ready();
        let (row, main_messages, mask_messages, kept_messages) =
            fold_fixture(session.plan(), 1);
        let mut prover = FoldProverSession::new(
            &session.plan().fold,
            session.plan().fold_layout.as_ref().unwrap(),
            &mask,
            row,
            main_messages,
            mask_messages,
            kept_messages,
        )
        .unwrap();
        let eps = session.epsilon().unwrap();
        let wrong = TestCoins::new(Digest([8; 32]));
        assert!(prover.start(eps, &wrong).is_err());
        assert_eq!(start.message.sum, prover.sum());
    }

    #[test]
    fn the_binder_binds_the_chain_random_bytes_and_its_run() {
        let (plan, _table) = plan_fixture();
        let parameters = plan.reader.configuration;
        let salt = Digest([0x77; 32]);
        let salted = FoldeqBinder::new_with_random_bytes(
            0x1234,
            parameters,
            &plan,
            Some(salt),
        )
        .expect("salted binder");
        let other = FoldeqBinder::new_with_random_bytes(
            0x1234,
            parameters,
            &plan,
            Some(Digest([0x78; 32])),
        )
        .expect("other salt");
        let unsalted = FoldeqBinder::new(0x1234, parameters, &plan).expect("unsalted");
        assert_ne!(salted.start(), other.start());
        assert_ne!(salted.start(), unsalted.start());
        assert_eq!(
            salted.run(),
            Some(
                crate::primitives::transcript::run_value(0x1234, &parameters, &salt)
                    .expect("run")
            )
        );
        assert_eq!(salted.random_bytes(), Some(&salt));
        assert_eq!(salted.chain().expect("chain").value(), salted.start());
        assert!(unsalted.run().is_none());
    }

}
