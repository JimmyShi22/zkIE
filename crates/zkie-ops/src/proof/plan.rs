//! Static model plans and plan checking for shared traces.
//!
//! A plan is constructed from model configuration and shapes only. It never
//! reads a proof and never executes the model. The verifier uses it to bind
//! every step, input port and final output to the expected data flow.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use zkie_core::artifact::digest_bytes;
use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;

use crate::op::OpCode;
use crate::proof::model_commitment::{
    AssetMeta, AssetOpening, ModelCommitmentPublic, SharedTrace, SharedTraceStep,
};
use crate::proof::relations::proof_output_values;

/// One expected input source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanSource {
    /// A previous step output, optionally sliced or transformed.
    Prev {
        step: usize,
        output: usize,
        layout: PlanLayout,
    },
    /// A model asset, optionally sliced or transformed.
    Asset { path: String, layout: PlanLayout },
    /// A trusted constant derived from public configuration.
    Constant(PlanConstant),
    /// Values concatenated from several sources.
    Concat(Vec<PlanSource>),
    /// Per-row concatenation of equally shaped head outputs.
    HeadConcat { parts: Vec<PlanSource>, rows: usize },
    /// Apply a layout to another source.
    Layout(Box<PlanSource>, PlanLayout),
}

impl PlanSource {
    pub fn prev(step: usize, output: usize) -> Self {
        Self::Prev {
            step,
            output,
            layout: PlanLayout::Flat,
        }
    }

    pub fn prev_slice(step: usize, output: usize, layout: PlanLayout) -> Self {
        Self::Prev {
            step,
            output,
            layout,
        }
    }

    pub fn asset(path: impl Into<String>) -> Self {
        Self::Asset {
            path: path.into(),
            layout: PlanLayout::Flat,
        }
    }

    pub fn asset_slice(path: impl Into<String>, layout: PlanLayout) -> Self {
        Self::Asset {
            path: path.into(),
            layout,
        }
    }

    pub fn with_layout(self, layout: PlanLayout) -> Self {
        Self::Layout(Box::new(self), layout)
    }
}

/// Static tensor layouts. Layouts are applied in order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlanLayout {
    Flat,
    Range {
        start: u64,
        count: usize,
    },
    Rows {
        row_width: usize,
        start: usize,
        count: usize,
    },
    Columns {
        row_width: usize,
        start: usize,
        count: usize,
    },
    Transpose {
        rows: usize,
        cols: usize,
    },
    Broadcast {
        rows: usize,
    },
    Chain(Vec<PlanLayout>),
}

impl PlanLayout {
    pub fn apply(&self, values: &[i64]) -> Result<Vec<i64>> {
        match self {
            Self::Flat => Ok(values.to_vec()),
            Self::Range { start, count } => {
                let start = usize::try_from(*start).map_err(|_| Error::InvalidIndex)?;
                let end = start.checked_add(*count).ok_or(Error::InvalidIndex)?;
                if end > values.len() {
                    return Err(Error::InvalidIndex);
                }
                Ok(values[start..end].to_vec())
            }
            Self::Rows {
                row_width,
                start,
                count,
            } => {
                if *row_width == 0 {
                    return Err(Error::ShapeMismatch);
                }
                let rows = values.len() / row_width;
                let end = start.checked_add(*count).ok_or(Error::InvalidIndex)?;
                if end > rows {
                    return Err(Error::InvalidIndex);
                }
                let begin = start
                    .checked_mul(*row_width)
                    .ok_or(Error::InvalidIndex)?;
                let finish = end
                    .checked_mul(*row_width)
                    .ok_or(Error::InvalidIndex)?;
                Ok(values[begin..finish].to_vec())
            }
            Self::Columns {
                row_width,
                start,
                count,
            } => {
                if *row_width == 0 {
                    return Err(Error::ShapeMismatch);
                }
                let end = start.checked_add(*count).ok_or(Error::InvalidIndex)?;
                if end > *row_width {
                    return Err(Error::InvalidIndex);
                }
                let rows = values.len() / row_width;
                let mut out = Vec::with_capacity(rows.saturating_mul(*count));
                for row in 0..rows {
                    let base = row * row_width + start;
                    out.extend_from_slice(&values[base..base + count]);
                }
                Ok(out)
            }
            Self::Transpose { rows, cols } => {
                if rows.checked_mul(*cols).ok_or(Error::ShapeMismatch)? != values.len() {
                    return Err(Error::ShapeMismatch);
                }
                let mut out = vec![0i64; values.len()];
                for row in 0..*rows {
                    for column in 0..*cols {
                        out[column * rows + row] = values[row * cols + column];
                    }
                }
                Ok(out)
            }
            Self::Broadcast { rows } => {
                let mut out = Vec::with_capacity(values.len().saturating_mul(*rows));
                for _ in 0..*rows {
                    out.extend_from_slice(values);
                }
                Ok(out)
            }
            Self::Chain(layouts) => {
                let mut current = values.to_vec();
                for layout in layouts {
                    current = layout.apply(&current)?;
                }
                Ok(current)
            }
        }
    }

    pub fn rows(row_width: usize, start: usize, count: usize) -> Self {
        Self::Rows {
            row_width,
            start,
            count,
        }
    }

    pub fn columns(row_width: usize, start: usize, count: usize) -> Self {
        Self::Columns {
            row_width,
            start,
            count,
        }
    }

    pub fn range(start: u64, count: usize) -> Self {
        Self::Range { start, count }
    }

    pub fn transpose(rows: usize, cols: usize) -> Self {
        Self::Transpose { rows, cols }
    }

    pub fn broadcast(rows: usize) -> Self {
        Self::Broadcast { rows }
    }

    pub fn chain(layouts: Vec<PlanLayout>) -> Self {
        Self::Chain(layouts)
    }
}

/// Trusted constants derived from public configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanConstant {
    Tokens,
    CausalMask {
        rows: usize,
        cols: usize,
        masked: i64,
    },
    VocabMask {
        rows: usize,
        vocab: usize,
        valid: usize,
        masked: i64,
    },
    Zeros {
        count: usize,
    },
    ScalarBroadcast {
        value: i64,
        count: usize,
    },
    Literal(Vec<i64>),
}

impl PlanConstant {
    fn materialize(&self, tokens: &[u32]) -> Result<Vec<i64>> {
        match self {
            Self::Tokens => Ok(tokens.iter().map(|value| *value as i64).collect()),
            Self::CausalMask { rows, cols, masked } => {
                if rows != cols {
                    return Err(Error::ShapeMismatch);
                }
                let mut values = vec![0i64; rows.checked_mul(*cols).ok_or(Error::ShapeMismatch)?];
                for row in 0..*rows {
                    for column in row + 1..*cols {
                        values[row * cols + column] = *masked;
                    }
                }
                Ok(values)
            }
            Self::VocabMask {
                rows,
                vocab,
                valid,
                masked,
            } => {
                if *valid > *vocab {
                    return Err(Error::ShapeMismatch);
                }
                let mut values = vec![0i64; rows.checked_mul(*vocab).ok_or(Error::ShapeMismatch)?];
                for row in 0..*rows {
                    for column in *valid..*vocab {
                        values[row * vocab + column] = *masked;
                    }
                }
                Ok(values)
            }
            Self::Zeros { count } => Ok(vec![0i64; *count]),
            Self::ScalarBroadcast { value, count } => Ok(vec![*value; *count]),
            Self::Literal(values) => Ok(values.clone()),
        }
    }
}

/// One plan configuration value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanConfigValue {
    Literal(i64),
    AssetI64 { path: String, index: usize },
}

/// One static step template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub opcode: OpCode,
    pub config: Vec<PlanConfigValue>,
    pub inputs: Vec<PlanSource>,
    pub outputs: usize,
}

/// Final output closure for an output-producing model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanFinalOutput {
    pub step: usize,
    pub output: usize,
}

/// The owner of a plan-level previous object: the producer step and an
/// identity derived from that step's plan description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviousObjectOwner {
    pub step: usize,
    pub id: Digest,
}

/// The tensor shape of a plan-level previous object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviousObjectShape {
    pub positions: usize,
    pub hidden: usize,
}

/// The position of a plan-level previous object in the consumer's flat input
/// tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviousObjectLayout {
    pub rows: usize,
    pub columns: usize,
    pub row_start: usize,
    pub column_start: usize,
}

/// A previous object declared by the trusted plan: the input matrix of the
/// selected final norm, produced by an earlier plan step and consumed by the
/// output-head segment. The fields are plan values; the verifier derives the
/// same object from the plan and compares it; no proof message contributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviousObjectPlan {
    /// Identity of the object, derived from the fields below.
    pub id: Digest,
    pub owner: PreviousObjectOwner,
    pub shape: PreviousObjectShape,
    pub layout: PreviousObjectLayout,
    /// The activation step whose input port reads the object.
    pub consumer_step: usize,
    pub consumer_port: usize,
    /// The producer's output port.
    pub producer_output: usize,
}

impl PreviousObjectPlan {
    /// Derives the object from the trusted plan and the selected activation
    /// step. The activation must be a final norm whose first input is the flat
    /// output of an earlier step; the object's shape is the norm's row/column
    /// shape and the consumer reads it at port 0 without a slice.
    pub fn derive_for_activation(
        plan: &TracePlan,
        activation_step: usize,
        positions: usize,
        hidden: usize,
    ) -> Result<Self> {
        if positions == 0 || hidden == 0 {
            return Err(Error::InvalidConfig);
        }
        positions
            .checked_mul(hidden)
            .ok_or(Error::ResourceLimit)?;
        let activation = plan
            .steps
            .get(activation_step)
            .ok_or(Error::StatementMismatch)?;
        if activation.opcode != OpCode::ProfileLayerNorm || activation.outputs == 0 {
            return Err(Error::StatementMismatch);
        }
        let config = literal_config_values(activation).ok_or(Error::InvalidConfig)?;
        if config.len() < 2
            || config[0] != positions as i64
            || config[1] != hidden as i64
        {
            return Err(Error::ShapeMismatch);
        }
        let source = activation
            .inputs
            .first()
            .ok_or(Error::ShapeMismatch)?;
        let (producer_step, producer_output) = match source {
            PlanSource::Prev {
                step,
                output,
                layout: PlanLayout::Flat,
            } => (*step, *output),
            _ => return Err(Error::StatementMismatch),
        };
        if producer_step == activation_step {
            return Err(Error::StatementMismatch);
        }
        let producer = plan
            .steps
            .get(producer_step)
            .ok_or(Error::StatementMismatch)?;
        if producer.outputs == 0 || producer_output >= producer.outputs {
            return Err(Error::ShapeMismatch);
        }
        let owner = PreviousObjectOwner {
            step: producer_step,
            id: owner_id(plan, producer_step),
        };
        let shape = PreviousObjectShape {
            positions,
            hidden,
        };
        let layout = PreviousObjectLayout {
            rows: positions,
            columns: hidden,
            row_start: 0,
            column_start: 0,
        };
        let mut object = Self {
            id: Digest::ZERO,
            owner,
            shape,
            layout,
            consumer_step: activation_step,
            consumer_port: 0,
            producer_output,
        };
        object.id = object.digest();
        Ok(object)
    }

    /// Checks a declared object against the plan-derived object.
    pub fn validate_for_activation(
        &self,
        plan: &TracePlan,
        activation_step: usize,
        positions: usize,
        hidden: usize,
    ) -> Result<()> {
        let expected =
            Self::derive_for_activation(plan, activation_step, positions, hidden)?;
        if *self != expected {
            return Err(Error::StatementMismatch);
        }
        Ok(())
    }

    fn digest(&self) -> Digest {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"zkie/previous-object/v1");
        bytes.extend_from_slice(&self.owner.id.0);
        put_word(&mut bytes, self.owner.step as u64);
        put_word(&mut bytes, self.shape.positions as u64);
        put_word(&mut bytes, self.shape.hidden as u64);
        put_word(&mut bytes, self.layout.rows as u64);
        put_word(&mut bytes, self.layout.columns as u64);
        put_word(&mut bytes, self.layout.row_start as u64);
        put_word(&mut bytes, self.layout.column_start as u64);
        put_word(&mut bytes, self.consumer_step as u64);
        put_word(&mut bytes, self.consumer_port as u64);
        put_word(&mut bytes, self.producer_output as u64);
        digest_bytes(&bytes)
    }
}

fn literal_config_values(step: &PlanStep) -> Option<Vec<i64>> {
    step.config
        .iter()
        .map(|value| match value {
            PlanConfigValue::Literal(value) => Some(*value),
            PlanConfigValue::AssetI64 { .. } => None,
        })
        .collect()
}

fn put_word(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    put_word(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn put_layout(bytes: &mut Vec<u8>, layout: &PlanLayout) {
    match layout {
        PlanLayout::Flat => put_word(bytes, 0),
        PlanLayout::Range { start, count } => {
            put_word(bytes, 1);
            put_word(bytes, *start);
            put_word(bytes, *count as u64);
        }
        PlanLayout::Rows {
            row_width,
            start,
            count,
        } => {
            put_word(bytes, 2);
            put_word(bytes, *row_width as u64);
            put_word(bytes, *start as u64);
            put_word(bytes, *count as u64);
        }
        PlanLayout::Columns {
            row_width,
            start,
            count,
        } => {
            put_word(bytes, 3);
            put_word(bytes, *row_width as u64);
            put_word(bytes, *start as u64);
            put_word(bytes, *count as u64);
        }
        PlanLayout::Transpose { rows, cols } => {
            put_word(bytes, 4);
            put_word(bytes, *rows as u64);
            put_word(bytes, *cols as u64);
        }
        PlanLayout::Broadcast { rows } => {
            put_word(bytes, 5);
            put_word(bytes, *rows as u64);
        }
        PlanLayout::Chain(layouts) => {
            put_word(bytes, 6);
            put_word(bytes, layouts.len() as u64);
            for layout in layouts {
                put_layout(bytes, layout);
            }
        }
    }
}

fn put_constant(bytes: &mut Vec<u8>, constant: &PlanConstant) {
    match constant {
        PlanConstant::Tokens => put_word(bytes, 0),
        PlanConstant::CausalMask {
            rows,
            cols,
            masked,
        } => {
            put_word(bytes, 1);
            put_word(bytes, *rows as u64);
            put_word(bytes, *cols as u64);
            put_word(bytes, *masked as u64);
        }
        PlanConstant::VocabMask {
            rows,
            vocab,
            valid,
            masked,
        } => {
            put_word(bytes, 2);
            put_word(bytes, *rows as u64);
            put_word(bytes, *vocab as u64);
            put_word(bytes, *valid as u64);
            put_word(bytes, *masked as u64);
        }
        PlanConstant::Zeros { count } => {
            put_word(bytes, 3);
            put_word(bytes, *count as u64);
        }
        PlanConstant::ScalarBroadcast { value, count } => {
            put_word(bytes, 4);
            put_word(bytes, *value as u64);
            put_word(bytes, *count as u64);
        }
        PlanConstant::Literal(values) => {
            put_word(bytes, 5);
            put_word(bytes, values.len() as u64);
            for value in values {
                put_word(bytes, *value as u64);
            }
        }
    }
}

fn put_source(bytes: &mut Vec<u8>, source: &PlanSource) {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout,
        } => {
            put_word(bytes, 0);
            put_word(bytes, *step as u64);
            put_word(bytes, *output as u64);
            put_layout(bytes, layout);
        }
        PlanSource::Asset { path, layout } => {
            put_word(bytes, 1);
            put_text(bytes, path);
            put_layout(bytes, layout);
        }
        PlanSource::Constant(constant) => {
            put_word(bytes, 2);
            put_constant(bytes, constant);
        }
        PlanSource::Concat(parts) => {
            put_word(bytes, 3);
            put_word(bytes, parts.len() as u64);
            for part in parts {
                put_source(bytes, part);
            }
        }
        PlanSource::HeadConcat { parts, rows } => {
            put_word(bytes, 4);
            put_word(bytes, parts.len() as u64);
            for part in parts {
                put_source(bytes, part);
            }
            put_word(bytes, *rows as u64);
        }
        PlanSource::Layout(inner, layout) => {
            put_word(bytes, 5);
            put_source(bytes, inner);
            put_layout(bytes, layout);
        }
    }
}

fn put_step(bytes: &mut Vec<u8>, step: &PlanStep) {
    put_word(bytes, step.opcode as u16 as u64);
    put_word(bytes, step.outputs as u64);
    put_word(bytes, step.config.len() as u64);
    for value in &step.config {
        match value {
            PlanConfigValue::Literal(value) => {
                put_word(bytes, 0);
                put_word(bytes, *value as u64);
            }
            PlanConfigValue::AssetI64 { path, index } => {
                put_word(bytes, 1);
                put_text(bytes, path);
                put_word(bytes, *index as u64);
            }
        }
    }
    put_word(bytes, step.inputs.len() as u64);
    for source in &step.inputs {
        put_source(bytes, source);
    }
}

fn owner_id(plan: &TracePlan, step_index: usize) -> Digest {
    let step = &plan.steps[step_index];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/previous-object-owner/v1");
    put_word(&mut bytes, plan.model_version as u64);
    put_word(&mut bytes, step_index as u64);
    put_step(&mut bytes, step);
    digest_bytes(&bytes)
}

/// A complete static trace plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TracePlan {
    pub steps: Vec<PlanStep>,
    pub argmax_steps: Vec<usize>,
    pub final_output: Option<PlanFinalOutput>,
    pub model_version: u32,
    /// The plan-level previous object of the output-head segment, when the
    /// plan has one. The verifier derives it from the plan and checks it.
    pub previous_object: Option<PreviousObjectPlan>,
}

impl TracePlan {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            argmax_steps: Vec::new(),
            final_output: None,
            model_version: 0,
            previous_object: None,
        }
    }

    /// Derives and records the previous object of an output-head plan whose
    /// activation is the final norm at `activation_step`.
    pub fn declare_previous_object_for_activation(
        &mut self,
        activation_step: usize,
        positions: usize,
        hidden: usize,
    ) -> Result<()> {
        let object =
            PreviousObjectPlan::derive_for_activation(self, activation_step, positions, hidden)?;
        self.previous_object = Some(object);
        Ok(())
    }
}

impl Default for TracePlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder used by model plan modules.
pub struct PlanBuilder {
    plan: TracePlan,
}

impl PlanBuilder {
    pub fn new() -> Self {
        Self {
            plan: TracePlan::new(),
        }
    }

    pub fn step(
        &mut self,
        opcode: OpCode,
        config: Vec<PlanConfigValue>,
        inputs: Vec<PlanSource>,
        outputs: usize,
    ) -> usize {
        let index = self.plan.steps.len();
        self.plan.steps.push(PlanStep {
            opcode,
            config,
            inputs,
            outputs,
        });
        index
    }

    pub fn argmax_step(&mut self, step: usize) {
        self.plan.argmax_steps.push(step);
    }

    pub fn final_output(&mut self, step: usize, output: usize) {
        self.plan.final_output = Some(PlanFinalOutput { step, output });
    }

    pub fn finish(self) -> TracePlan {
        self.plan
    }
}

impl Default for PlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn config(values: Vec<i64>) -> Vec<PlanConfigValue> {
    values.into_iter().map(PlanConfigValue::Literal).collect()
}

/// Identity of an authenticated asset, bound to the model version and anchor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AssetCacheKey {
    model_version: u32,
    anchor_id: Digest,
    model_commitment: Digest,
    asset_id: u64,
    path: String,
    start: Digest,
    root: Digest,
    leaf_root: Digest,
    element_count: u64,
    rows: u64,
}

/// Assets whose materialized values are reused by the trusted plan.
pub(crate) type AssetCacheKeys = BTreeSet<(String, PlanLayout)>;

pub(crate) fn cache_keys(plan: &TracePlan) -> AssetCacheKeys {
    let mut counts: BTreeMap<(String, PlanLayout), usize> = BTreeMap::new();
    for step in &plan.steps {
        for source in &step.inputs {
            count_source(source, &mut counts);
        }
        for value in &step.config {
            if let PlanConfigValue::AssetI64 { path, .. } = value {
                *counts.entry((path.clone(), PlanLayout::Flat)).or_insert(0) += 1;
            }
        }
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(key, _)| key)
        .collect()
}

fn count_source(source: &PlanSource, counts: &mut BTreeMap<(String, PlanLayout), usize>) {
    match source {
        PlanSource::Asset { path, layout } => {
            *counts.entry((path.clone(), layout.clone())).or_insert(0) += 1;
        }
        PlanSource::Concat(parts) => {
            for part in parts {
                count_source(part, counts);
            }
        }
        PlanSource::HeadConcat { parts, .. } => {
            for part in parts {
                count_source(part, counts);
            }
        }
        PlanSource::Layout(inner, _) => count_source(inner, counts),
        PlanSource::Prev { .. } | PlanSource::Constant(_) => {}
    }
}

pub(crate) struct AssetView<'a> {
    public: &'a ModelCommitmentPublic,
    trace: &'a SharedTrace,
    model_version: u32,
    cache_keys: AssetCacheKeys,
    full_cache: RefCell<BTreeMap<AssetCacheKey, Rc<Vec<i64>>>>,
    slice_cache: RefCell<BTreeMap<(AssetCacheKey, PlanLayout), Rc<Vec<i64>>>>,
}

impl<'a> AssetView<'a> {
    /// The caller verifies every opening before constructing the view; cached
    /// values are only derived from those verified openings and are keyed by
    /// the model version, anchor identity and full asset identity.
    pub(crate) fn new(
        public: &'a ModelCommitmentPublic,
        trace: &'a SharedTrace,
        plan: &TracePlan,
    ) -> Self {
        Self {
            public,
            trace,
            model_version: plan.model_version,
            cache_keys: cache_keys(plan),
            full_cache: RefCell::new(BTreeMap::new()),
            slice_cache: RefCell::new(BTreeMap::new()),
        }
    }

    fn meta(&self, path: &str) -> Result<&AssetMeta> {
        self.public
            .assets
            .iter()
            .find(|asset| asset.path == path)
            .ok_or(Error::AssetMismatch)
    }

    fn opening(&self, path: &str) -> Result<&AssetOpening> {
        let meta = self.meta(path)?;
        self.trace
            .assets
            .iter()
            .find(|opening| opening.asset_id == meta.asset_id)
            .ok_or(Error::StatementMismatch)
    }

    fn cache_key(&self, path: &str) -> Result<AssetCacheKey> {
        let meta = self.meta(path)?;
        Ok(AssetCacheKey {
            model_version: self.model_version,
            anchor_id: self.public.anchor.anchor_id,
            model_commitment: self.public.anchor.model_commitment,
            asset_id: meta.asset_id,
            path: meta.path.clone(),
            start: meta.start,
            root: meta.root,
            leaf_root: meta.leaf_record.leaf_root,
            element_count: meta.element_count,
            rows: meta.rows,
        })
    }

    fn cacheable(&self, path: &str, layout: &PlanLayout) -> bool {
        self.cache_keys
            .contains(&(path.to_owned(), layout.clone()))
    }

    fn full_values(&self, path: &str) -> Result<Rc<Vec<i64>>> {
        let key = self.cache_key(path)?;
        let cacheable = self.cacheable(path, &PlanLayout::Flat);
        if cacheable {
            if let Some(values) = self.full_cache.borrow().get(&key) {
                return Ok(Rc::clone(values));
            }
        }
        let values = Rc::new(self.materialize_full(path)?);
        if cacheable {
            self.full_cache
                .borrow_mut()
                .insert(key, Rc::clone(&values));
        }
        Ok(values)
    }

    fn materialize_full(&self, path: &str) -> Result<Vec<i64>> {
        let meta = self.meta(path)?;
        let opening = self.opening(path)?;
        if opening.row_count != meta.rows || opening.rows.len() != meta.rows as usize {
            return Err(Error::IncompleteArtifact);
        }
        let count = usize::try_from(meta.element_count).map_err(|_| Error::ResourceLimit)?;
        let mut values = Vec::with_capacity(count);
        for row in &opening.rows {
            for value in &row.values {
                if values.len() == count {
                    break;
                }
                values.push(*value as i64);
            }
            if values.len() == count {
                break;
            }
        }
        if values.len() != count {
            return Err(Error::IncompleteArtifact);
        }
        Ok(values)
    }

    fn source_values(&self, path: &str, layout: &PlanLayout) -> Result<Rc<Vec<i64>>> {
        if matches!(layout, PlanLayout::Flat) {
            return self.full_values(path);
        }
        let key = self.cache_key(path)?;
        let cacheable = self.cacheable(path, layout);
        if cacheable {
            if let Some(values) = self
                .slice_cache
                .borrow()
                .get(&(key.clone(), layout.clone()))
            {
                return Ok(Rc::clone(values));
            }
        }
        let full = self.full_values(path)?;
        let values = Rc::new(layout.apply(&full)?);
        if cacheable {
            self.slice_cache
                .borrow_mut()
                .insert((key, layout.clone()), Rc::clone(&values));
        }
        Ok(values)
    }

    fn asset_id(&self, path: &str) -> Result<u64> {
        Ok(self.meta(path)?.asset_id)
    }

    fn config_values(&self, values: &[PlanConfigValue]) -> Result<Vec<i64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            match value {
                PlanConfigValue::Literal(value) => out.push(*value),
                PlanConfigValue::AssetI64 { path, index } => {
                    let asset = self.full_values(path)?;
                    out.push(*asset.get(*index).ok_or(Error::InvalidIndex)?);
                }
            }
        }
        Ok(out)
    }
}

enum Materialized<'a> {
    Slice(&'a [i64]),
    Owned(Rc<Vec<i64>>),
}

impl Materialized<'_> {
    fn as_slice(&self) -> &[i64] {
        match self {
            Self::Slice(values) => values,
            Self::Owned(values) => values.as_slice(),
        }
    }
}

impl std::ops::Deref for Materialized<'_> {
    type Target = [i64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

fn materialize<'a>(
    source: &PlanSource,
    verified_outputs: &'a [Vec<Vec<i64>>],
    assets: &AssetView<'_>,
    tokens: &[u32],
) -> Result<Materialized<'a>> {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout,
        } => {
            let values = verified_outputs
                .get(*step)
                .and_then(|outputs| outputs.get(*output))
                .ok_or(Error::InvalidIndex)?;
            if matches!(layout, PlanLayout::Flat) {
                Ok(Materialized::Slice(values))
            } else {
                Ok(Materialized::Owned(Rc::new(layout.apply(values)?)))
            }
        }
        PlanSource::Asset { path, layout } => {
            Ok(Materialized::Owned(assets.source_values(path, layout)?))
        }
        PlanSource::Constant(constant) => Ok(Materialized::Owned(Rc::new(
            constant.materialize(tokens)?,
        ))),
        PlanSource::Concat(parts) => {
            let mut out = Vec::new();
            for part in parts {
                let part = materialize(part, verified_outputs, assets, tokens)?;
                out.extend_from_slice(&part);
            }
            Ok(Materialized::Owned(Rc::new(out)))
        }
        PlanSource::HeadConcat { parts, rows } => {
            let mut values = Vec::with_capacity(parts.len());
            for part in parts {
                values.push(materialize(part, verified_outputs, assets, tokens)?);
            }
            Ok(Materialized::Owned(Rc::new(head_concat(&values, *rows)?)))
        }
        PlanSource::Layout(inner, layout) => {
            if matches!(layout, PlanLayout::Flat) {
                return materialize(inner, verified_outputs, assets, tokens);
            }
            let values = materialize(inner, verified_outputs, assets, tokens)?;
            Ok(Materialized::Owned(Rc::new(layout.apply(&values)?)))
        }
    }
}

/// Per-row concatenation of equally shaped parts. Any length disagreement is
/// rejected before slicing, so a short part cannot index out of bounds.
fn head_concat(parts: &[Materialized<'_>], rows: usize) -> Result<Vec<i64>> {
    if rows == 0 || parts.is_empty() {
        return Err(Error::ShapeMismatch);
    }
    let len = parts[0].len();
    if len % rows != 0 || parts.iter().any(|part| part.len() != len) {
        return Err(Error::ShapeMismatch);
    }
    let head_dim = len / rows;
    let mut out = Vec::with_capacity(parts.len() * len);
    for row in 0..rows {
        for part in parts {
            out.extend_from_slice(&part[row * head_dim..(row + 1) * head_dim]);
        }
    }
    Ok(out)
}

enum SourceKind {
    Prev {
        step: usize,
        output: usize,
        flat: bool,
    },
    Asset,
    Other,
}

fn source_kind(source: &PlanSource) -> SourceKind {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout,
        } => SourceKind::Prev {
            step: *step,
            output: *output,
            flat: matches!(layout, PlanLayout::Flat),
        },
        PlanSource::Asset { .. } => SourceKind::Asset,
        PlanSource::Constant(_) => SourceKind::Other,
        PlanSource::Concat(_) => SourceKind::Other,
        PlanSource::HeadConcat { .. } => SourceKind::Other,
        PlanSource::Layout(inner, _) => source_kind(inner),
    }
}

fn external_flat_path(source: &PlanSource) -> Option<&str> {
    match source {
        PlanSource::Asset {
            path,
            layout: PlanLayout::Flat,
        } => Some(path.as_str()),
        _ => None,
    }
}

/// Check the step structure against the trusted plan. This runs before any
/// relation verification consumes proof ports, so port counts, config and
/// external/source maps are trusted-plan constrained first.
pub(crate) fn verify_step_structure(
    step: &SharedTraceStep,
    plan_step: &PlanStep,
    assets: &AssetView<'_>,
) -> Result<()> {
    if plan_step.opcode as u16 != step.message.opcode {
        return Err(Error::StatementMismatch);
    }
    if assets.config_values(&plan_step.config)? != step.message.config {
        return Err(Error::StatementMismatch);
    }
    let input_count = step.message.proof.lengths.inputs.len();
    if input_count != plan_step.inputs.len()
        || step.message.proof.inputs.len() != input_count
        || step.external.len() != input_count
        || step.sources.len() != input_count
    {
        return Err(Error::ShapeMismatch);
    }
    let output_count = step.message.proof.lengths.outputs.len();
    if output_count != plan_step.outputs || step.message.proof.outputs.len() != output_count {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}

pub(crate) fn verify_step(
    step: &SharedTraceStep,
    plan_step: &PlanStep,
    verified_outputs: &[Vec<Vec<i64>>],
    assets: &AssetView<'_>,
    tokens: &[u32],
) -> Result<Vec<Vec<i64>>> {
    verify_step_structure(step, plan_step, assets)?;
    let output_count = step.message.proof.lengths.outputs.len();
    let mut inputs = Vec::with_capacity(plan_step.inputs.len());
    for (port, source) in plan_step.inputs.iter().enumerate() {
        let external = step.external.get(port).copied().flatten();
        let edge = step.sources.get(port).copied().flatten();
        if matches!(source_kind(source), SourceKind::Asset) && external.is_some() {
            if edge.is_some() {
                return Err(Error::StatementMismatch);
            }
            if let Some(path) = external_flat_path(source) {
                let pool = external.ok_or(Error::StatementMismatch)?;
                let meta = assets.meta(path)?;
                // A flat external asset is supplied by the verified opening
                // itself. Check the planned identity and full coverage; the
                // values are materialized from the same opening below, so no
                // prover-supplied copy enters.
                if meta.row_width != 0 && meta.element_count % meta.row_width as u64 == 0 {
                    let opening = assets
                        .trace
                        .assets
                        .get(pool)
                        .ok_or(Error::StatementMismatch)?;
                    if opening.asset_id != meta.asset_id
                        || opening.start != meta.start
                        || opening.root != meta.root
                        || opening.row_count != meta.rows
                        || opening.rows.len() != meta.rows as usize
                    {
                        return Err(Error::IncompleteArtifact);
                    }
                }
            }
        }
        let expected = materialize(source, verified_outputs, assets, tokens)?;
        match source_kind(source) {
            SourceKind::Prev {
                step,
                output,
                flat,
            } => {
                if external.is_some() {
                    return Err(Error::StatementMismatch);
                }
                if flat {
                    if edge != Some((step, output)) {
                        return Err(Error::StatementMismatch);
                    }
                } else if edge.is_some() {
                    return Err(Error::StatementMismatch);
                }
            }
            SourceKind::Asset => {
                if edge.is_some() {
                    return Err(Error::StatementMismatch);
                }
                if let Some(pool) = external {
                    let plan_asset = assets.asset_id(match source {
                        PlanSource::Asset { path, .. } => path,
                        PlanSource::Layout(inner, _) => match inner.as_ref() {
                            PlanSource::Asset { path, .. } => path,
                            _ => return Err(Error::StatementMismatch),
                        },
                        _ => return Err(Error::StatementMismatch),
                    })?;
                    let opening = assets
                        .trace
                        .assets
                        .get(pool)
                        .ok_or(Error::StatementMismatch)?;
                    if opening.asset_id != plan_asset {
                        return Err(Error::StatementMismatch);
                    }
                }
            }
            SourceKind::Other => {
                if external.is_some() || edge.is_some() {
                    return Err(Error::StatementMismatch);
                }
            }
        }
        inputs.push(expected.as_slice().to_vec());
    }
    let outputs = proof_output_values(&step.message.proof)?;
    if outputs.len() != output_count {
        return Err(Error::ShapeMismatch);
    }
    Ok(inputs)
}

/// Version of the shared-trace input binding. Version 2 materializes every
/// relation input from its trusted plan source and carries no input openings.
pub const SHARED_TRACE_INPUT_VERSION: u64 = 2;

/// A shared-trace step carries no input openings. A step that still carries
/// one was produced by the version 1 form and is rejected.
pub(crate) fn check_shared_input_form(step: &SharedTraceStep) -> Result<()> {
    if step.message.proof.inputs.iter().any(|group| !group.is_empty()) {
        return Err(Error::UnsupportedVersion);
    }
    Ok(())
}

pub(crate) fn verify_outputs(
    trace: &SharedTrace,
    plan: &TracePlan,
    verified_outputs: &[Vec<Vec<i64>>],
    assets: &AssetView<'_>,
) -> Result<()> {
    let mut argmax = Vec::with_capacity(plan.argmax_steps.len());
    for step in &plan.argmax_steps {
        let plan_step = plan.steps.get(*step).ok_or(Error::StatementMismatch)?;
        if plan_step.opcode != OpCode::Argmax || plan_step.inputs.is_empty() {
            return Err(Error::StatementMismatch);
        }
        let input = materialize(
            &plan_step.inputs[0],
            verified_outputs,
            assets,
            &trace.tokens,
        )?;
        if input.is_empty() {
            return Err(Error::InvalidConfig);
        }
        let mut best = 0usize;
        for index in 1..input.len() {
            if input[index] > input[best] {
                best = index;
            }
        }
        let output = verified_outputs
            .get(*step)
            .and_then(|outputs| outputs.first())
            .and_then(|values| values.first())
            .ok_or(Error::StatementMismatch)?;
        if *output != best as i64 {
            return Err(Error::InvalidProof);
        }
        argmax.push(*output as u32);
    }
    if trace.argmax != argmax {
        return Err(Error::StatementMismatch);
    }
    match &plan.final_output {
        Some(final_output) => {
            let output = verified_outputs
                .get(final_output.step)
                .and_then(|outputs| outputs.get(final_output.output))
                .ok_or(Error::StatementMismatch)?;
            if trace.output != *output {
                return Err(Error::StatementMismatch);
            }
        }
        None => {
            if !trace.output.is_empty() {
                return Err(Error::StatementMismatch);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_keys_select_only_reused_sources() {
        let mut builder = PlanBuilder::new();
        builder.step(
            OpCode::TableLookup,
            Vec::new(),
            vec![
                PlanSource::asset("reused.bin"),
                PlanSource::asset("once.bin"),
            ],
            1,
        );
        builder.step(
            OpCode::TableLookup,
            Vec::new(),
            vec![
                PlanSource::asset("reused.bin"),
                PlanSource::prev(0, 0),
            ],
            1,
        );
        let plan = builder.finish();
        let keys = cache_keys(&plan);
        assert!(keys.contains(&("reused.bin".to_owned(), PlanLayout::Flat)));
        assert!(!keys.contains(&("once.bin".to_owned(), PlanLayout::Flat)));
    }

    #[test]
    fn head_concat_rejects_mismatched_part_lengths() {
        let first = vec![1i64, 2, 3, 4];
        let short = vec![5i64, 6];
        let parts = [Materialized::Slice(&first), Materialized::Slice(&short)];
        assert!(matches!(head_concat(&parts, 2), Err(Error::ShapeMismatch)));

        let long = vec![5i64, 6, 7, 8, 9, 10];
        let parts = [Materialized::Slice(&first), Materialized::Slice(&long)];
        assert!(matches!(head_concat(&parts, 2), Err(Error::ShapeMismatch)));

        let second = vec![5i64, 6, 7, 8];
        let parts = [Materialized::Slice(&first), Materialized::Slice(&second)];
        assert_eq!(
            head_concat(&parts, 2).unwrap(),
            vec![1, 2, 5, 6, 3, 4, 7, 8]
        );
    }
}
