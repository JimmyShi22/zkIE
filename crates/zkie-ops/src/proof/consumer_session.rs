//! Production assembly of one request's consumer chain.
//!
//! One request registers the real projection steps that consume the encoded
//! table, derives the request plan from the trusted model plan and the
//! validated trace values, and proves the first, shared and second stages over
//! the same matrix. The prover side reads the asset mapping one stored row at a
//! time and keeps only one encoded table panel; the verifier side rebuilds the
//! same plans from public facts and the committed table metadata, opens nothing
//! of the asset, and reads only the request proof file.
//!
//! The request-dependent plan structure (the head obligations, the table
//! identity, the mask allocation, the shared later plan and the integer
//! bounds) is derived from the trusted sources on both sides. The values the
//! verifier cannot derive (the plan member, its target, the group mask root and
//! the committed raw and bit roots) travel in the proof file and are bound by
//! the first chain and the group; the verifier never trusts them as structure.

use std::path::Path;
use std::sync::Arc;

use zkie_core::artifact::digest_bytes;
use zkie_core::common::field::PrimeField64;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::pcs::request::field::{Fp3, Fp};
use zkie_engine::pcs::request::fold::{ByteMaskCoins, MaskCoins};
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_engine::pcs::request::runtime_tree::MaskTreePlan;
use zkie_engine::pcs::request::table::{commit_table_streaming, IndexReader, IndexShape};
use zkie_engine::primitives::codec::CanonicalWriter;

use crate::op::OpCode;
use crate::proof::asset_table::{
    asset_row_source, verify_asset_table_meta, AssetTableInput, AssetTableMeta, AssetTableParams,
};
use crate::proof::column_session::{column_ledger_dir, AssetTableKeeper};
use crate::proof::consumer_first::{
    derive_consumer_first_member, prove_consumer_first, ConsumerFirstDraft,
    ConsumerFirstGroupWitness, ConsumerFirstMessages, ConsumerFirstPlan, ConsumerWCommitRandomness,
};
use crate::proof::consumer_group::{
    commit_consumer_group_mask, derive_consumer_group_shapes, ConsumerGroupEntries,
    ConsumerGroupPlan,
};
use crate::proof::consumer_plan::ConsumerPlanAssembly;
use crate::proof::consumer_proof::{ConsumerProofFile, ConsumerProofTotals};
use crate::proof::consumer_psc::{
    ConsumerMaskSource, ConsumerObligation, ConsumerPscMasks, ConsumerPscShape, ConsumerTable,
};
use crate::proof::consumer_raw::{
    consumer_raw_mask_entries, ConsumerRawFirst, ConsumerRawGroup, ConsumerRawMaskPlan,
    ConsumerRawMaskWitness, ConsumerRawPlan, ConsumerRawShape,
};
use crate::proof::consumer_request::{
    consumer_activation_digest, consumer_outputs_digest, check_raw_rounding_witness,
    raw_rounding_remainder, signed_fp3, ConsumerIntegerBound, ConsumerIntegerRanges,
    ConsumerRequestPlan, ConsumerSharedPlan,
};
use crate::proof::consumer_second::{
    consumer_second_chain, prove_consumer_own, ConsumerChainBinding, ConsumerOwnPlan,
    ConsumerOwnWitness,
};
use crate::proof::consumer_shared::{
    consumer_binding_prove, consumer_shared_charge, prove_consumer_shared, verify_consumer_shared,
    ConsumerSharedGeometry, ConsumerSharedIds, ConsumerSharedStage,
};
use crate::proof::consumer_source::{AssetRange, ConsumerMatrixSource};
use crate::proof::hidden_member::PendingRow;
use crate::proof::hidden_send_session::{
    commit_request_w, decode_shared_messages, derive_w_values, pending_row_weights,
    HiddenTableProver, WShape,
};
use crate::proof::lin_binding::LinBindingShape;
use crate::proof::model_commitment::{
    AssetCommit, AssetReader, AssetStore, ModelAssetSpec, ModelCommitment,
};
use crate::proof::output_closure::{commit_value_group, CommittedValueGroup};
use crate::proof::plan::{PlanConfigValue, PlanLayout, PlanSource, TracePlan};
use crate::proof::projection_claim::ConsumerMatrixColumns;

/// The request identifier of the single request a proof carries today.
pub const CONSUMER_REQUEST_ID: u64 = 1;
/// The public oracle identifier of the consumer table in the request statement.
pub const CONSUMER_TABLE_ORACLE: u64 = 30;
/// The consumer mask message oracle. Every PSC, eval and raw mask slot of the
/// request lives in this one message.
pub const CONSUMER_MASK_ORACLE: u64 = 3000;
/// The PSC block starts at the first mask slot.
pub const CONSUMER_PSC_HIDING_BASE: u64 = 0;
/// The W message identifiers.
pub const CONSUMER_W_ORACLE: u64 = 3001;
pub const CONSUMER_W_GROUP: u64 = 43;
pub const CONSUMER_RAW_GROUP: u64 = 41;
pub const CONSUMER_BITS_GROUP: u64 = 42;
const RAW_ORACLE_BASE: u64 = 1000;
const BITS_ORACLE_BASE: u64 = 2000;
const GROUP_ID: u64 = 4000;
const GROUP_FRESH: u64 = 4001;
const GROUP_TREE_BASE: u64 = 5000;
const GROUP_MASK_TREE: u64 = 5100;
const GU_ORACLE: u64 = 6000;
const OMEGA_ORACLE: u64 = 6001;
const GU_TREE: u64 = 6002;
const OMEGA_TREE: u64 = 6003;
const SHARED_MASK_TREE: u64 = 6004;
const SHARED_FRESH: u64 = 6005;
const SHARED_BASE_POSITIONS: usize = 1;
const OWN_MASK_TREE: u64 = 6100;
const OWN_FRESH: u64 = 6101;
const LEAF_SALT_BYTES: usize = 40;

/// The private request seed. One seed is drawn per request and split by
/// domain into the raw, mask, W, binding, shared and own streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerRequestSeed {
    digest: Digest,
}

impl ConsumerRequestSeed {
    pub fn from_digest(digest: Digest) -> Result<Self> {
        if digest == Digest::ZERO {
            return Err(Error::InvalidConfig);
        }
        Ok(Self { digest })
    }

    /// Draws a fresh private seed. It is never persisted and never enters the
    /// proof; every commitment and every run stream is derived from it.
    pub fn sample() -> Result<Self> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|_| Error::EntropyUnavailable)?;
        Self::from_digest(Digest(bytes))
    }

    fn stream(&self, label: &[u8]) -> Result<ByteMaskCoins> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(label);
        writer.put_digest(&self.digest);
        ByteMaskCoins::new(digest_bytes(&writer.into_bytes()))
    }

    fn digest(&self, label: &[u8]) -> Result<Digest> {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(b"consumer-request-seed-stream");
        writer.put_bytes(label);
        writer.put_digest(&self.digest);
        Ok(digest_bytes(&writer.into_bytes()))
    }
}

/// The real request witness: the activation values of the activation step, the
/// raw accumulators of every registered head slice and the public requantized
/// output rows of those same steps.
pub struct ConsumerRequestWitness<'a> {
    /// The activation rows, `positions * records` signed words.
    pub activation: &'a [i64],
    /// One raw accumulator block per head, `positions * head_width` entries.
    pub raw: &'a [Vec<i128>],
    /// One public output row block per head, in the same order.
    pub outputs: &'a [Vec<i64>],
}

/// The trusted table binding both sides derive from the committed metadata.
#[derive(Clone, Debug)]
pub struct ConsumerTableBinding {
    pub path: String,
    pub oracle: u64,
    pub root: Digest,
    pub identity: Digest,
    pub configuration: Digest,
    pub shape: IndexShape,
    pub shared: ConsumerSharedPlan,
    pub ranges: ConsumerIntegerRanges,
}

impl ConsumerTableBinding {
    /// Reads the binding of one committed table from its metadata and the
    /// derived table parameters. The metadata must already be authenticated by
    /// the anchor; the derived identity and configuration are compared here.
    pub fn from_meta(
        meta: &AssetTableMeta,
        params: &AssetTableParams,
        table_model_identity: Digest,
        spec: &ModelAssetSpec,
        shared: ConsumerSharedPlan,
    ) -> Result<Self> {
        verify_asset_table_meta(meta, table_model_identity, spec, params)?;
        // The request binds the committed reader (root, configuration, shape
        // and record), which is the identity the consumer stages check. The
        // plan-level table identity above stays the anchor-side declaration.
        let reader = consumer_table_reader(meta, params, table_model_identity, spec)?;
        let identity = reader_table_identity(&reader)?;
        Ok(Self {
            path: params.path.clone(),
            oracle: CONSUMER_TABLE_ORACLE,
            root: meta.root,
            identity,
            configuration: meta.configuration,
            shape: meta.shape,
            shared,
            ranges: ConsumerIntegerRanges {
                x_abs_bound: 1,
                weight_lo: params.valid_min,
                weight_hi: params.valid_max,
            },
        })
    }

    fn consumer_table(&self) -> ConsumerTable {
        ConsumerTable {
            path: self.path.clone(),
            oracle: self.oracle,
            root: self.root,
            identity: self.identity,
            configuration: self.configuration,
        }
    }
}

/// One request's derived plan structure, before the prover-bound values are
/// fixed.
#[derive(Clone, Debug)]
pub struct ConsumerRequestAssembly {
    pub request: ConsumerRequestPlan,
    pub raw: ConsumerRawShape,
    pub mask: ConsumerMaskSource,
    pub raw_hiding_base: u64,
    pub entries: ConsumerGroupEntries,
    pub run: Digest,
    pub configuration: Digest,
}

fn plan_domain(label: &[u8], base: &Digest) -> Digest {
    let mut writer = CanonicalWriter::new();
    writer.put_bytes(b"consumer-plan-domain-v1");
    writer.put_bytes(label);
    writer.put_digest(base);
    digest_bytes(&writer.into_bytes())
}

struct RegisteredStep {
    source_step: usize,
    activation_step: u32,
    head: u32,
    records: u64,
    places: u64,
    shift: i64,
    slice: (u64, u64),
    positions: u32,
}

/// The registered projection steps of the trusted model plan. Every step that
/// consumes the table's head slice is one obligation; the steps must be the
/// contiguous head order `0..J` of one activation step and one geometry.
fn registered_steps(plan: &TracePlan, table: &ConsumerTableBinding) -> Result<Vec<RegisteredStep>> {
    let mut steps = Vec::new();
    for (index, step) in plan.steps.iter().enumerate() {
        if step.opcode != OpCode::Projection || step.config.len() != 5 || step.inputs.len() != 2 {
            continue;
        }
        let (path, layout) = match &step.inputs[1] {
            PlanSource::Asset { path, layout } | PlanSource::ConsumerAsset { path, layout } => {
                (path, layout)
            }
            _ => continue,
        };
        let PlanLayout::Columns {
            row_width,
            start,
            count,
        } = layout
        else {
            continue;
        };
        let (row_width, start, count) = (*row_width, *start, *count);
        if path != &table.path {
            continue;
        }
        let mut literals = [0i64; 5];
        for (slot, value) in literals.iter_mut().zip(&step.config) {
            match value {
                PlanConfigValue::Literal(value) => *slot = *value,
                PlanConfigValue::AssetI64 { .. } => return Err(Error::InvalidConfig),
            }
        }
        let positions = u32::try_from(literals[0]).map_err(|_| Error::InvalidConfig)?;
        let width = u64::try_from(count).map_err(|_| Error::InvalidConfig)?;
        if width == 0
            || literals[1] != i64::try_from(width).map_err(|_| Error::InvalidConfig)?
            || literals[4] != 0
        {
            return Err(Error::InvalidConfig);
        }
        let head = u64::try_from(start).map_err(|_| Error::InvalidConfig)?;
        if head % width != 0 {
            return Err(Error::InvalidConfig);
        }
        let activation_step = match &step.inputs[0] {
            PlanSource::Prev {
                step: at,
                output: 0,
                layout: PlanLayout::Flat,
            } => u32::try_from(*at).map_err(|_| Error::InvalidConfig)?,
            _ => return Err(Error::InvalidConfig),
        };
        steps.push(RegisteredStep {
            source_step: index,
            activation_step,
            head: u32::try_from(head / width).map_err(|_| Error::InvalidConfig)?,
            records: u64::try_from(literals[2]).map_err(|_| Error::InvalidConfig)?,
            places: u64::try_from(row_width).map_err(|_| Error::InvalidConfig)?,
            shift: literals[3],
            slice: (head, head + width),
            positions,
        });
    }
    if steps.is_empty() {
        return Err(Error::UnsupportedPrimitive);
    }
    let first = &steps[0];
    for (expected, step) in steps.iter().enumerate() {
        if step.head as usize != expected
            || step.activation_step != first.activation_step
            || step.positions != first.positions
            || step.records != first.records
            || step.places != first.places
            || step.shift != first.shift
            || step.slice.1 - step.slice.0 != first.slice.1 - first.slice.0
        {
            return Err(Error::StatementMismatch);
        }
    }
    Ok(steps)
}

/// Derives the request plan structure from the trusted model plan, the
/// committed table binding and the validated request values.
pub fn derive_consumer_request_assembly(
    plan: &TracePlan,
    table: &ConsumerTableBinding,
    activation: &[i64],
    outputs: &[Vec<i64>],
) -> Result<ConsumerRequestAssembly> {
    let registered = registered_steps(plan, table)?;
    if outputs.len() != registered.len() {
        return Err(Error::ShapeMismatch);
    }
    let first = &registered[0];
    let head_width = first.slice.1 - first.slice.0;
    let positions = first.positions;
    let real = usize::try_from(positions)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(usize::try_from(head_width).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if outputs.iter().any(|row| row.len() != real) {
        return Err(Error::ShapeMismatch);
    }
    let activation_len = usize::try_from(positions)
        .map_err(|_| Error::ResourceLimit)?
        .checked_mul(usize::try_from(first.records).map_err(|_| Error::ResourceLimit)?)
        .ok_or(Error::ResourceLimit)?;
    if activation.len() != activation_len {
        return Err(Error::ShapeMismatch);
    }
    if first.places < head_width * u64::from(registered.len() as u32) {
        return Err(Error::InvalidConfig);
    }
    let shape = ConsumerPscShape {
        positions,
        places: first.places,
        records: first.records,
        slice: (0, head_width * u64::from(registered.len() as u32)),
    };
    let hiding_entries = shape.hiding_entries()?;
    let mask = ConsumerMaskSource {
        oracle: CONSUMER_MASK_ORACLE,
        hiding_base: CONSUMER_PSC_HIDING_BASE,
        shown_slot: CONSUMER_PSC_HIDING_BASE + hiding_entries as u64,
    };
    let raw_hiding_base = mask.shown_slot + 1;
    let consumer_table = table.consumer_table();
    let mut obligations = Vec::with_capacity(registered.len());
    for step in &registered {
        let plan_step = plan
            .steps
            .get(step.source_step)
            .ok_or(Error::StatementMismatch)?;
        obligations.push(ConsumerObligation::from_projection_step(
            CONSUMER_REQUEST_ID,
            u32::try_from(step.source_step).map_err(|_| Error::ResourceLimit)?,
            step.activation_step,
            step.head,
            plan_step,
            consumer_table.clone(),
            mask,
        )?);
    }
    let activation_fp3 = signed_words(activation)?;
    let activation_digest = consumer_activation_digest(&activation_fp3);
    let ranges = ConsumerIntegerRanges {
        x_abs_bound: activation_abs_bound(activation)?,
        weight_lo: table.ranges.weight_lo,
        weight_hi: table.ranges.weight_hi,
    };
    let request = ConsumerRequestPlan::new(
        CONSUMER_REQUEST_ID,
        consumer_table.clone(),
        mask,
        table.shared,
        ranges,
        activation_digest,
        Digest([0x42; 32]),
        obligations.clone(),
    )?;
    let outputs_digest = consumer_outputs_digest(&request, outputs)?;
    let request = ConsumerRequestPlan::new(
        CONSUMER_REQUEST_ID,
        consumer_table,
        mask,
        table.shared,
        ranges,
        activation_digest,
        outputs_digest,
        obligations,
    )?;
    let heads = request.heads();
    let raw_columns = u64::from(positions)
        .checked_mul(head_width)
        .ok_or(Error::ResourceLimit)?;
    let shift = u32::try_from(first.shift).map_err(|_| Error::InvalidConfig)?;
    let bits_columns = raw_columns
        .checked_mul(u64::from(shift))
        .ok_or(Error::ResourceLimit)?;
    let raw = ConsumerRawShape {
        heads,
        positions,
        width: head_width,
        shift,
        columns: raw_columns,
    };
    raw.fault()
        .map_or(Ok(()), |_| Err(Error::InvalidConfig))?;
    let mask_entries = consumer_raw_mask_entries(&raw, raw_hiding_base)?
        .max(mask.shown_slot + 1)
        .max(mask.hiding_base + hiding_entries as u64);
    let entries = ConsumerGroupEntries {
        heads,
        raw_columns,
        bits_columns,
        w_columns: request.shape.places,
        mask_entries,
        rows: 6 * u64::from(heads) + 5,
    };
    let base = request.statement_digest();
    Ok(ConsumerRequestAssembly {
        request,
        raw,
        mask,
        raw_hiding_base,
        entries,
        run: plan_domain(b"consumer-group-run", &base),
        configuration: plan_domain(b"consumer-group-configuration", &base),
    })
}

fn signed_words(values: &[i64]) -> Result<Vec<Fp3>> {
    values.iter().map(|value| signed_fp3(i128::from(*value))).collect()
}

fn activation_abs_bound(values: &[i64]) -> Result<u64> {
    if values.is_empty() {
        return Err(Error::InvalidConfig);
    }
    let mut bound = 0u64;
    for value in values {
        bound = bound.max(value.unsigned_abs());
    }
    if bound == 0 {
        return Err(Error::InvalidConfig);
    }
    Ok(bound)
}

/// The committed plans of one request. Both sides call this one function; only
/// the provenance of the committed values differs.
pub fn consumer_first_plan(
    assembly: &ConsumerRequestAssembly,
    table: &ConsumerTableBinding,
    values: &ConsumerPlanAssembly,
    w_root: Digest,
) -> Result<(ConsumerFirstPlan, ConsumerGroupPlan)> {
    let shapes = derive_consumer_group_shapes(
        assembly.entries,
        assembly.run,
        assembly.configuration,
        GROUP_TREE_BASE,
        GROUP_MASK_TREE,
    )?;
    let raw = ConsumerRawPlan {
        shape: assembly.raw,
        raw: ConsumerRawGroup {
            shape: shapes.raw_shape,
            columns: assembly.raw.columns,
            group: CONSUMER_RAW_GROUP,
            oracle_base: RAW_ORACLE_BASE,
            openings: 1,
            roots: values.raw_roots.clone(),
        },
        bits: ConsumerRawGroup {
            shape: shapes.bits_shape,
            columns: assembly.raw.bits_columns().ok_or(Error::InvalidConfig)?,
            group: CONSUMER_BITS_GROUP,
            oracle_base: BITS_ORACLE_BASE,
            openings: 1,
            roots: values.bits_roots.clone(),
        },
        mask: ConsumerRawMaskPlan {
            oracle: CONSUMER_MASK_ORACLE,
            base_slot: assembly.raw_hiding_base,
        },
        outputs_digest: assembly.request.outputs_digest,
        bound: ConsumerIntegerBound::new(
            assembly.request.shape.records,
            assembly.request.ranges,
            assembly.raw.shift,
        )?,
    };
    raw.validate()?;
    let draft = ConsumerFirstDraft {
        request: assembly.request.clone(),
        raw,
        w_shape: shapes.w_shape,
        w_oracle: CONSUMER_W_ORACLE,
        w_group: CONSUMER_W_GROUP,
        table_shape: table.shape,
        matrix_rows: assembly.request.shape.places,
        k_lo: 0,
        columns: assembly.request.shape.places,
    };
    draft.check()?;
    let heads = assembly.request.heads();
    let group = ConsumerGroupPlan {
        shapes,
        group_id: GROUP_ID,
        fresh: GROUP_FRESH,
        raw_oracle_base: RAW_ORACLE_BASE,
        bits_oracle_base: BITS_ORACLE_BASE,
        mask_oracle: CONSUMER_MASK_ORACLE,
        w_oracle: CONSUMER_W_ORACLE,
        raw_bat_base: u64::from(heads),
        bits_bat_base: 0,
        mask_bat: 2 * u64::from(heads),
        w_bat: 2 * u64::from(heads) + 1,
        mask_root: values.mask_root,
        raw_roots: values.raw_roots.clone(),
        bits_roots: values.bits_roots.clone(),
        w_roots: vec![w_root],
    };
    group.check()?;
    let plan = ConsumerFirstPlan::new(draft, values.member.clone(), values.target)?
        .with_group(group.clone())?;
    Ok((plan, group))
}

/// The streaming matrix over one committed asset: `row(record)` reads the
/// stored asset row `record` and exposes `M[place, record] = A[record, place]`.
pub struct ConsumerAssetMatrix<'a> {
    store: &'a AssetStore<'a>,
    path: String,
    places: u64,
    records: u64,
}

impl<'a> ConsumerAssetMatrix<'a> {
    pub fn new(store: &'a AssetStore<'a>, path: &str, places: u64, records: u64) -> Result<Self> {
        if places == 0 || records == 0 {
            return Err(Error::InvalidConfig);
        }
        store.pool(path)?;
        Ok(Self {
            store,
            path: path.to_string(),
            places,
            records,
        })
    }
}

impl ConsumerMatrixColumns for ConsumerAssetMatrix<'_> {
    fn places(&self) -> u64 {
        self.places
    }

    fn records(&self) -> u64 {
        self.records
    }

    fn row(&self, record: u64, out: &mut [Fp3]) -> Result<()> {
        if record >= self.records || out.len() as u64 != self.places {
            return Err(Error::ShapeMismatch);
        }
        let start = record.checked_mul(self.places).ok_or(Error::ResourceLimit)?;
        let values = self.store.value_range(
            &self.path,
            start,
            usize::try_from(self.places).map_err(|_| Error::ResourceLimit)?,
        )?;
        for (slot, value) in out.iter_mut().zip(&values) {
            *slot = signed_fp3(i128::from(*value))?;
        }
        Ok(())
    }
}

fn draw_nonzero(coins: &mut ByteMaskCoins) -> Result<Fp3> {
    let mut value = Fp3::ZERO;
    while value == Fp3::ZERO {
        value = coins.draw()?.0;
    }
    Ok(value)
}

fn many_salts(coins: &mut ByteMaskCoins, count: u64) -> Result<Vec<Vec<u8>>> {
    let mut salts = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut salt = Vec::new();
        while salt.len() < LEAF_SALT_BYTES {
            let (value, _) = coins.draw()?;
            for coefficient in value.c {
                salt.extend_from_slice(&coefficient.as_canonical_u64().to_le_bytes());
            }
        }
        salt.truncate(LEAF_SALT_BYTES);
        salts.push(salt);
    }
    Ok(salts)
}

fn commit_group_values(
    shape: &WShape,
    oracle_base: u64,
    group: u64,
    values: &[Vec<Fp3>],
    coins: &mut ByteMaskCoins,
) -> Result<(CommittedValueGroup, Vec<Vec<Fp3>>)> {
    let positions = values.len();
    let answers = shape.answers as usize;
    let mut pads = Vec::with_capacity(positions * answers);
    for _ in 0..positions * answers {
        pads.push(draw_nonzero(coins)?);
    }
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let mut tails = Vec::with_capacity(positions);
    for _ in 0..positions {
        let mut tail = Vec::with_capacity(rows * tail_len);
        for _ in 0..rows * tail_len {
            tail.push(coins.draw()?.0);
        }
        tails.push(tail);
    }
    let mut salts = Vec::with_capacity(positions);
    for _ in 0..positions {
        salts.push(many_salts(coins, shape.code.n())?);
    }
    let committed = commit_value_group(
        shape,
        values
            .first()
            .map(|row| row.len() as u64)
            .ok_or(Error::ShapeMismatch)?,
        oracle_base,
        group,
        values,
        &pads,
        &tails,
        &salts,
        u32::try_from(positions).map_err(|_| Error::ResourceLimit)?,
    )?;
    Ok((committed, tails))
}

fn commit_request_randomness(
    shape: &WShape,
    coins: &mut ByteMaskCoins,
) -> Result<ConsumerWCommitRandomness> {
    let mut pads = Vec::with_capacity(shape.answers as usize);
    for _ in 0..shape.answers {
        pads.push(draw_nonzero(coins)?);
    }
    let rows = usize::try_from(shape.main_rows).map_err(|_| Error::ResourceLimit)?;
    let tail_len = usize::try_from(shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let mut flat = Vec::with_capacity(rows * tail_len);
    for _ in 0..rows * tail_len {
        flat.push(coins.draw()?.0);
    }
    let randomness = ConsumerWCommitRandomness {
        pads,
        tails: vec![flat],
        salts: vec![many_salts(coins, shape.code.n())?],
    };
    randomness.check(shape)?;
    Ok(randomness)
}

fn split_tails(flat: &[Fp3], rows: usize, tail_len: usize) -> Result<Vec<Vec<Fp3>>> {
    if flat.len() != rows.checked_mul(tail_len).ok_or(Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    Ok(flat.chunks_exact(tail_len).map(|tail| tail.to_vec()).collect())
}

fn spec_of(asset: &AssetCommit) -> ModelAssetSpec {
    ModelAssetSpec {
        asset_id: asset.asset_id,
        path: asset.path.clone(),
        element_count: asset.element_count,
        row_width: asset.row_width,
        encoding: asset.encoding.clone(),
    }
}

/// The prover-side table keeper of the encoded table: the streaming row source
/// and the column budget of the real asset.
pub fn open_consumer_table_keeper(
    commitment: &ModelCommitment,
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    input: &AssetTableInput,
    columns: u64,
    max_columns: u64,
    max_bytes: u64,
) -> Result<AssetTableKeeper> {
    if input.params != *params {
        return Err(Error::StatementMismatch);
    }
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.path == params.path)
        .ok_or(Error::AssetMismatch)?;
    let spec = spec_of(asset);
    verify_asset_table_meta(meta, table_model_identity, &spec, params)?;
    let window = asset
        .element_count
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(1 << 30))
        .ok_or(Error::ResourceLimit)?;
    let governor = BudgetedMemoryGovernor::new(window);
    let reader = Arc::new(AssetReader::open(commitment, asset.asset_id, &governor)?);
    let mut source = asset_row_source(Arc::clone(&reader), &spec, params, input.randomness.tails.clone())?;
    let panel_columns = params.shape.code.n().min(1024).max(1);
    let table = commit_table_streaming(
        &mut source,
        params.shape,
        meta.configuration,
        &input.randomness.salts,
        panel_columns,
        max_bytes,
    )?;
    if table.reader.root != meta.root || table.reader.record != meta.record {
        return Err(Error::StatementMismatch);
    }
    let ledger_dir = column_ledger_dir(Path::new(&commitment.root), meta.root);
    AssetTableKeeper::open_with_source(
        table,
        Box::new(source),
        ledger_dir,
        columns,
        max_columns,
    )
}

/// The verifier-side reader of the committed table: the metadata alone, no
/// asset row and no encoded table.
pub fn consumer_table_reader(
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    spec: &ModelAssetSpec,
) -> Result<IndexReader> {
    verify_asset_table_meta(meta, table_model_identity, spec, params)?;
    let reader = IndexReader {
        shape: meta.shape,
        configuration: meta.configuration,
        root: meta.root,
        record: meta.record.clone(),
    };
    if reader.fault().is_some() {
        return Err(Error::StatementMismatch);
    }
    Ok(reader)
}

fn shared_geometry(
    first_plan: &ConsumerFirstPlan,
    table: &ConsumerTableBinding,
    reader: &IndexReader,
) -> Result<ConsumerSharedGeometry> {
    let parameters = {
        let mut writer = CanonicalWriter::new();
        writer.put_bytes(b"consumer-request-parameters-v1");
        writer.put_digest(&first_plan.draft.request.statement_digest());
        writer.put_digest(&table.identity);
        writer.put_digest(&table.configuration);
        writer.put_digest(&table.root);
        writer.put_u64(first_plan.draft.columns);
        writer.put_u64(first_plan.draft.matrix_rows);
        writer.put_u64(u64::from(first_plan.draft.k_lo));
        digest_bytes(&writer.into_bytes())
    };
    ConsumerSharedGeometry::derive(
        first_plan,
        table.identity,
        table.configuration,
        table.shape,
        LinBindingShape {
            log_leaves: 0,
            salt_bytes: LEAF_SALT_BYTES,
        },
        plan_domain(b"consumer-binding-configuration", &parameters),
        parameters,
        first_plan.draft.request.statement_bytes(),
        reader.name(),
        ConsumerSharedIds {
            group: GROUP_ID,
            gu_oracle: GU_ORACLE,
            omega_oracle: OMEGA_ORACLE,
            gu_tree: GU_TREE,
            omega_tree: OMEGA_TREE,
            mask_tree: SHARED_MASK_TREE,
            fresh: SHARED_FRESH,
            base_positions: SHARED_BASE_POSITIONS,
            column_work_bits: 0,
        },
    )
}

fn binding_salts(seed: &ConsumerRequestSeed) -> Result<Vec<u8>> {
    let mut coins = seed.stream(b"consumer-binding")?;
    Ok(many_salts(&mut coins, 1)?.remove(0))
}

/// Proves one request's consumer chain over the real committed table.
#[allow(clippy::too_many_arguments)]
pub fn prove_consumer_request(
    plan: &TracePlan,
    commitment: &ModelCommitment,
    store: &AssetStore<'_>,
    tables: &[AssetTableInput],
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    table_model_identity: Digest,
    shared: ConsumerSharedPlan,
    witness: &ConsumerRequestWitness<'_>,
    seed: ConsumerRequestSeed,
) -> Result<ConsumerProofFile> {
    if witness.raw.len() != witness.outputs.len() || witness.raw.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let asset = commitment
        .assets
        .iter()
        .find(|asset| asset.path == params.path)
        .ok_or(Error::AssetMismatch)?;
    let spec = spec_of(asset);
    let table = ConsumerTableBinding::from_meta(
        meta,
        params,
        table_model_identity,
        &spec,
        shared,
    )?;
    let assembly = derive_consumer_request_assembly(plan, &table, witness.activation, witness.outputs)?;
    let raw_columns = usize::try_from(assembly.raw.columns).map_err(|_| Error::ResourceLimit)?;
    for (raw, outputs) in witness.raw.iter().zip(witness.outputs) {
        if raw.len() != raw_columns
            || outputs.len() != raw_columns
            || outputs.len() != usize::try_from(assembly.request.shape.positions)
                .map_err(|_| Error::ResourceLimit)?
                * usize::try_from(assembly.raw.width).map_err(|_| Error::ResourceLimit)?
        {
            return Err(Error::ShapeMismatch);
        }
        check_raw_rounding_witness(raw, outputs, assembly.raw.shift)?;
    }
    let activation = signed_words(witness.activation)?;
    let raw_values: Vec<Vec<Fp3>> = witness
        .raw
        .iter()
        .map(|head| head.iter().map(|value| signed_fp3(*value)).collect::<Result<_>>())
        .collect::<Result<_>>()?;
    let mut bits_values: Vec<Vec<Fp3>> = Vec::with_capacity(witness.raw.len());
    for (raw, outputs) in witness.raw.iter().zip(witness.outputs) {
        let mut bits = vec![Fp3::ZERO; raw_columns * assembly.raw.shift as usize];
        for (entry, (value, output)) in raw.iter().zip(outputs).enumerate() {
            let remainder = raw_rounding_remainder(*value, *output, assembly.raw.shift)?;
            for bit in 0..assembly.raw.shift as usize {
                if (remainder >> bit) & 1 == 1 {
                    bits[entry * assembly.raw.shift as usize + bit] = Fp3::ONE;
                }
            }
        }
        bits_values.push(bits);
    }
    let shapes = derive_consumer_group_shapes(
        assembly.entries,
        assembly.run,
        assembly.configuration,
        GROUP_TREE_BASE,
        GROUP_MASK_TREE,
    )?;
    let mut raw_coins = seed.stream(b"consumer-raw-values")?;
    let (raw_group, raw_tails) = commit_group_values(
        &shapes.raw_shape,
        RAW_ORACLE_BASE,
        CONSUMER_RAW_GROUP,
        &raw_values,
        &mut raw_coins,
    )?;
    let mut bits_coins = seed.stream(b"consumer-bit-values")?;
    let (bits_group, bits_tails) = commit_group_values(
        &shapes.bits_shape,
        BITS_ORACLE_BASE,
        CONSUMER_BITS_GROUP,
        &bits_values,
        &mut bits_coins,
    )?;
    let raw_first = ConsumerRawFirst::from_groups(&raw_group.group, &bits_group.group);
    let raw_probe = ConsumerRawPlan {
        shape: assembly.raw,
        raw: ConsumerRawGroup {
            shape: shapes.raw_shape,
            columns: assembly.raw.columns,
            group: CONSUMER_RAW_GROUP,
            oracle_base: RAW_ORACLE_BASE,
            openings: 1,
            roots: raw_group.roots.clone(),
        },
        bits: ConsumerRawGroup {
            shape: shapes.bits_shape,
            columns: assembly.raw.bits_columns().ok_or(Error::InvalidConfig)?,
            group: CONSUMER_BITS_GROUP,
            oracle_base: BITS_ORACLE_BASE,
            openings: 1,
            roots: bits_group.roots.clone(),
        },
        mask: ConsumerRawMaskPlan {
            oracle: CONSUMER_MASK_ORACLE,
            base_slot: assembly.raw_hiding_base,
        },
        outputs_digest: assembly.request.outputs_digest,
        bound: ConsumerIntegerBound::new(
            assembly.request.shape.records,
            assembly.request.ranges,
            assembly.raw.shift,
        )?,
    };
    raw_probe.validate()?;
    let spaces = raw_probe.mask_spaces()?;
    let mut mask_coins = seed.stream(b"consumer-mask-values")?;
    let hiding_entries = assembly.request.shape.hiding_entries()?;
    let mut psc_hiding = Vec::with_capacity(hiding_entries);
    for _ in 0..hiding_entries {
        psc_hiding.push(draw_nonzero(&mut mask_coins)?);
    }
    let shown = draw_nonzero(&mut mask_coins)?;
    let psc_masks = ConsumerPscMasks::new(psc_hiding.clone(), shown);
    let mut raw_masks = ConsumerRawMaskWitness::default();
    for space in &spaces {
        for slot in &space.hiding {
            let value = draw_nonzero(&mut mask_coins)?;
            raw_masks.insert(*slot, value);
        }
        let value = draw_nonzero(&mut mask_coins)?;
        raw_masks.insert(space.shown, value);
    }
    let mask_total = usize::try_from(shapes.mask_message_entries()?).map_err(|_| Error::ResourceLimit)?;
    let mask_ell = usize::try_from(shapes.first_mask.ell).map_err(|_| Error::ResourceLimit)?;
    if mask_ell == 0 || mask_total % mask_ell != 0 {
        return Err(Error::ShapeMismatch);
    }
    if mask_total < usize::try_from(assembly.entries.mask_entries).map_err(|_| Error::ResourceLimit)? {
        return Err(Error::ShapeMismatch);
    }
    let mut mask = vec![Fp3::ZERO; mask_total];
    for (index, value) in psc_hiding.iter().enumerate() {
        mask[assembly.mask.hiding_base as usize + index] = *value;
    }
    mask[assembly.mask.shown_slot as usize] = shown;
    for space in &spaces {
        for slot in &space.hiding {
            mask[*slot as usize] = raw_masks.entry(*slot)?;
        }
        mask[space.shown as usize] = raw_masks.entry(space.shown)?;
    }
    let mask_rows: Vec<Vec<Fp3>> = mask.chunks(mask_ell).map(|row| row.to_vec()).collect();
    let mut mask_tails = Vec::with_capacity(mask_rows.len());
    for _ in 0..mask_rows.len() {
        let mut tail = Vec::with_capacity(usize::try_from(shapes.first_mask.r).map_err(|_| Error::ResourceLimit)?);
        for _ in 0..shapes.first_mask.r {
            tail.push(mask_coins.draw()?.0);
        }
        mask_tails.push(tail);
    }
    let mask_salts = many_salts(&mut mask_coins, shapes.first_mask.n())?;
    let placeholder = ConsumerGroupPlan {
        shapes: shapes.clone(),
        group_id: GROUP_ID,
        fresh: GROUP_FRESH,
        raw_oracle_base: RAW_ORACLE_BASE,
        bits_oracle_base: BITS_ORACLE_BASE,
        mask_oracle: CONSUMER_MASK_ORACLE,
        w_oracle: CONSUMER_W_ORACLE,
        raw_bat_base: u64::from(assembly.request.heads()),
        bits_bat_base: 0,
        mask_bat: 2 * u64::from(assembly.request.heads()),
        w_bat: 2 * u64::from(assembly.request.heads()) + 1,
        mask_root: Digest([0x11; 32]),
        raw_roots: raw_group.roots.clone(),
        bits_roots: bits_group.roots.clone(),
        w_roots: vec![Digest([0x12; 32])],
    };
    let mask_commit =
        commit_consumer_group_mask(&placeholder, mask_rows.clone(), mask_tails.clone(), mask_salts)?;
    let draft = ConsumerFirstDraft {
        request: assembly.request.clone(),
        raw: raw_probe,
        w_shape: shapes.w_shape,
        w_oracle: CONSUMER_W_ORACLE,
        w_group: CONSUMER_W_GROUP,
        table_shape: table.shape,
        matrix_rows: assembly.request.shape.places,
        k_lo: 0,
        columns: assembly.request.shape.places,
    };
    draft.check()?;
    let mut matrix = ConsumerAssetMatrix::new(
        store,
        &table.path,
        assembly.request.shape.places,
        assembly.request.shape.records,
    )?;
    let (member, target) = derive_consumer_first_member(
        &draft,
        &activation,
        &raw_values,
        &bits_values,
        witness.outputs,
        &raw_first,
        &raw_masks,
        &mut matrix,
        &psc_masks,
    )?;
    let weights = pending_row_weights(&member, assembly.request.shape.places)?;
    let mut w_source = ConsumerMatrixSource::new(AssetRange::new(
        store,
        &table.path,
        usize::try_from(assembly.request.shape.places).map_err(|_| Error::ResourceLimit)?,
        assembly.request.shape.records,
    )?)?;
    let w_values = derive_w_values(
        &table.shape,
        assembly.request.shape.places,
        0,
        assembly.request.shape.places,
        &weights,
        &mut w_source,
    )?;
    let mut w_coins = seed.stream(b"consumer-w-values")?;
    let w_randomness = commit_request_randomness(&draft.w_shape, &mut w_coins)?;
    let w_commit = commit_request_w(
        &draft.w_shape,
        CONSUMER_W_ORACLE,
        CONSUMER_W_GROUP,
        &w_values,
        &w_randomness.pads,
        &w_randomness.tails,
        &w_randomness.salts,
    )?;
    let w_root = *w_commit.roots.first().ok_or(Error::ShapeMismatch)?;
    let values = ConsumerPlanAssembly {
        member,
        target,
        mask_root: mask_commit.root,
        raw_roots: raw_group.roots.clone(),
        bits_roots: bits_group.roots.clone(),
    };
    let (first_plan, _group) = consumer_first_plan(&assembly, &table, &values, w_root)?;
    let group_witness = ConsumerFirstGroupWitness {
        raw: &raw_group,
        raw_tails: &raw_tails,
        bits: &bits_group,
        bits_tails: &bits_tails,
        mask: &mask_commit,
        mask_rows: mask_rows.clone(),
        mask_tails: mask_tails.clone(),
        w_tails: &w_randomness.tails,
        run_seed: seed.digest(b"consumer-group-run-seed")?,
    };
    let mut table_source = ConsumerMatrixSource::new(AssetRange::new(
        store,
        &table.path,
        usize::try_from(assembly.request.shape.places).map_err(|_| Error::ResourceLimit)?,
        assembly.request.shape.records,
    )?)?;
    let (messages, _obligations) = prove_consumer_first(
        &first_plan,
        &activation,
        &raw_values,
        &bits_values,
        witness.outputs,
        &raw_first,
        &raw_masks,
        &psc_masks,
        &mut matrix,
        &mut table_source,
        &w_randomness,
        Some(&group_witness),
    )?;
    let salts = binding_salts(&seed)?;
    let input = tables
        .iter()
        .find(|input| input.params.path == params.path)
        .ok_or(Error::MissingCommitArtifact)?;
    let reader = consumer_table_reader(meta, params, table_model_identity, &spec)?;
    let geometry = shared_geometry(&first_plan, &table, &reader)?;
    let (binding_path, binding_root) = consumer_binding_prove(
        &messages.first_seal,
        geometry.binding_shape,
        &geometry.binding_configuration,
        salts,
    )?;
    let stage = ConsumerSharedStage::new(&shared)?;
    let mut keeper = open_consumer_table_keeper(
        commitment,
        meta,
        params,
        table_model_identity,
        input,
        stage.columns.batch_columns,
        stage.columns.max_columns,
        u64::MAX,
    )?;
    let charge = consumer_shared_charge(&stage, &mut keeper)?;
    let shared_proof = prove_consumer_shared(
        &geometry,
        &stage,
        &reader,
        &mut keeper,
        seed.digest(b"consumer-shared-run")?,
        &charge,
        &binding_root,
    )?;
    let check_stage = ConsumerSharedStage::new(&shared)?;
    let shared_verified = verify_consumer_shared(
        &geometry,
        &check_stage,
        &reader,
        &shared_proof.messages,
        &binding_root,
    )?;
    let own_mask_tree = MaskTreePlan {
        tree_number: OWN_MASK_TREE,
        leaf_class: 4,
        log_n: geometry.mask_code.log_n,
        mask_rows: 1,
        salt_bytes: LEAF_SALT_BYTES,
        run: plan_domain(b"consumer-own-mask-run", &geometry.parameters),
        configuration: plan_domain(b"consumer-own-mask-config", &geometry.parameters),
    };
    let own_plan = ConsumerOwnPlan::from_first_shared(
        &first_plan,
        &messages,
        &geometry,
        &shared_verified,
        own_mask_tree,
        OWN_FRESH,
        1,
    )?;
    let chain_binding = ConsumerChainBinding::from_first_shared(&messages, &shared_verified)?;
    let chain = consumer_second_chain(&geometry, &chain_binding)?;
    let tail_rows = usize::try_from(first_plan.draft.w_shape.main_rows)
        .map_err(|_| Error::ResourceLimit)?;
    let tail_len =
        usize::try_from(first_plan.draft.w_shape.code.r).map_err(|_| Error::ResourceLimit)?;
    let w_tails = split_tails(&w_randomness.tails[0], tail_rows, tail_len)?;
    let own_witness = ConsumerOwnWitness {
        w: &w_commit,
        w_tails: &w_tails,
        shared: &shared_proof.witness,
    };
    let own_proof = prove_consumer_own(
        chain,
        &own_plan,
        &shared_verified,
        &own_witness,
        seed.digest(b"consumer-own-run")?,
    )?;
    let totals = ConsumerProofTotals {
        first_seal: messages.first_seal,
        binding_root,
        shared_seal: shared_proof.messages.seal,
        second_seal: own_proof.seal,
        eval_target: messages.eval_target,
        w_root: messages.w_root,
        omega_root: shared_proof.messages.omega_root,
    };
    ConsumerProofFile::assemble_with_totals(
        &first_plan,
        &messages,
        &shared_proof.messages,
        &own_proof,
        totals,
        binding_path,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_engine::pcs::request::table::TableDistances;

    fn entries(positions: u32, shift: u32, mask_entries: u64) -> ConsumerGroupEntries {
        let raw_columns = u64::from(positions) * 64;
        ConsumerGroupEntries {
            heads: 12,
            raw_columns,
            bits_columns: raw_columns * u64::from(shift),
            w_columns: 1024,
            mask_entries,
            rows: 76,
        }
    }

    /// The production entry counts derive one group at both sequence lengths,
    /// and the mask member covers every allocated mask slot.
    #[test]
    fn the_production_group_entries_derive_at_both_sequence_lengths() {
        for (positions, expected_mask) in [(16u32, 685u64), (512, 925)] {
            let shape = ConsumerRawShape {
                heads: 12,
                positions,
                width: 64,
                shift: 16,
                columns: u64::from(positions) * 64,
            };
            let mask_entries = consumer_raw_mask_entries(&shape, 25).unwrap();
            assert_eq!(mask_entries, expected_mask);
            let shapes = derive_consumer_group_shapes(
                entries(positions, 16, mask_entries),
                Digest([0x61; 32]),
                Digest([0x62; 32]),
                5000,
                5100,
            )
            .unwrap();
            assert!(shapes.mask_message_entries().unwrap() >= mask_entries);
            assert_eq!(shapes.entries.rows, 76);
            let _ = TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 284,
                m_later: 284,
            };
        }
    }
}

/// Verifies one request's consumer chain from public facts and the committed
/// table metadata. The verifier reads no asset row and trusts no plan
/// structure from the proof: it rebuilds the same plans and then runs the full
/// first, shared and second chain in the global order.
#[allow(clippy::too_many_arguments)]
pub fn verify_consumer_request(
    plan: &TracePlan,
    trace_outputs: &[Vec<Vec<i64>>],
    table_model_identity: Digest,
    meta: &AssetTableMeta,
    params: &AssetTableParams,
    spec: &ModelAssetSpec,
    shared: ConsumerSharedPlan,
    file: &ConsumerProofFile,
) -> Result<crate::proof::consumer_second::ConsumerTotalAccepted> {
    let (_, w_root) = ConsumerFirstMessages::peek_header(&file.first)?;
    let table = ConsumerTableBinding::from_meta(meta, params, table_model_identity, spec, shared)?;
    let registered = registered_steps(plan, &table)?;
    let first = &registered[0];
    let activation_step = first.activation_step as usize;
    let activation = trace_outputs
        .get(activation_step)
        .and_then(|tensors| tensors.first())
        .cloned()
        .ok_or(Error::StatementMismatch)?;
    let mut outputs = Vec::with_capacity(registered.len());
    for step in &registered {
        let row = trace_outputs
            .get(step.source_step)
            .and_then(|tensors| tensors.first())
            .cloned()
            .ok_or(Error::StatementMismatch)?;
        outputs.push(row);
    }
    let assembly = derive_consumer_request_assembly(plan, &table, &activation, &outputs)?;
    let activation_fp3 = signed_words(&activation)?;
    let values = file.plan_assembly()?;
    let (first_plan, _group) = consumer_first_plan(&assembly, &table, &values, w_root)?;
    let first_messages = ConsumerFirstMessages::decode(&first_plan, &file.first)?;
    let reader = consumer_table_reader(meta, params, table_model_identity, spec)?;
    let geometry = shared_geometry(&first_plan, &table, &reader)?;
    let shared_messages = decode_shared_messages(&file.shared)?;
    let own_mask_tree = MaskTreePlan {
        tree_number: OWN_MASK_TREE,
        leaf_class: 4,
        log_n: geometry.mask_code.log_n,
        mask_rows: 1,
        salt_bytes: LEAF_SALT_BYTES,
        run: plan_domain(b"consumer-own-mask-run", &geometry.parameters),
        configuration: plan_domain(b"consumer-own-mask-config", &geometry.parameters),
    };
    let own_plan = ConsumerOwnPlan::from_first_omega(
        &first_plan,
        &first_messages,
        &geometry,
        shared_messages.omega_root,
        own_mask_tree,
        OWN_FRESH,
        1,
    )?;
    let stage = ConsumerSharedStage::new(&shared)?;
    crate::proof::consumer_proof::verify_consumer_chain(
        &first_plan,
        &activation_fp3,
        &outputs,
        &geometry,
        &stage,
        &reader,
        &own_plan,
        file,
    )
}
