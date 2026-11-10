//! Isolated two-step graph-edge demonstration.
//!
//! This module verifies the selected `ProfileLayerNorm` step `4443`, compares
//! its authenticated output with the authenticated input-0 opening of the
//! selected `MatMul` step `4444`, and binds a typed producer/consumer edge into
//! the candidate column-session first query.
//!
//! It is not a complete graph verifier and does not accept a proof. Step
//! `4443` executes the real `ProfileLayerNorm` relation and authenticates its
//! openings; step `4444` only authenticates input-0, not the consumer relation.
//! The upstream input of step `4443` remains a boundary; no step before `4443`
//! is verified. The real selected `MatMul` control remains the separate 2c
//! baseline. The edge comparison is a local value comparison after the two
//! relevant opening/relation checks; it is not proximity or a complete graph
//! closure. The edge declaration distinguishes previous-step edges (no asset
//! identity) from asset-reference edges (asset identity encoded).

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::foldeq::{
    AssetEdge, EdgeDirection, EdgeLayout, FoldeqEdgeBinding, PrevEdge,
};

use crate::op::OpCode;
use crate::proof::binding::{verify_opening, OracleId, OracleRegistry, OracleOpening};
use crate::proof::column_session::{
    verify_column_session, ColumnBindingInput, ColumnSessionOutcome,
};
use crate::proof::model_commitment::{
    verify_model_anchor, AssetMeta, AssetOpening, ModelCommitmentPublic, SharedTrace,
    SharedTraceStep,
};
use crate::proof::plan::{PlanConfigValue, PlanLayout, PlanSource, TracePlan};
use crate::proof::relations::{verify_op_relation_with_external, OpRelationPublic};
use crate::proof::session_baseline::TrustedModel;

/// Result of the producer relation check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProducerRelation {
    pub step_index: usize,
    pub output: Vec<i64>,
    pub output_root: Digest,
    pub output_oracle: OracleId,
    pub asset_roots: Vec<(usize, Digest)>,
}

/// Result of authenticating the consumer input group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerInput {
    pub step_index: usize,
    pub port: usize,
    pub values: Vec<i64>,
    pub root: Digest,
    pub oracle: OracleId,
}

/// The complete two-step edge demonstration result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeSessionOutcome {
    pub edge: FoldeqEdgeBinding,
    pub producer_step: usize,
    pub consumer_step: usize,
    pub input_port: usize,
    pub producer_output: Vec<i64>,
    pub consumer_input: Vec<i64>,
    pub producer_root: Digest,
    pub consumer_root: Digest,
    pub relation_verified: bool,
    pub input_authenticated: bool,
    pub values_equal: bool,
    pub column_session: ColumnSessionOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SourceKind {
    Prev { step: usize, output: usize },
    Asset { path: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceSpec {
    kind: SourceKind,
    layout: EdgeLayout,
}

fn edge_layout(layout: &PlanLayout) -> Result<EdgeLayout> {
    match layout {
        PlanLayout::Flat => Ok(EdgeLayout::Flat),
        PlanLayout::Range { start, count } => Ok(EdgeLayout::Range {
            start: *start,
            count: u64::try_from(*count).map_err(|_| Error::ResourceLimit)?,
        }),
        PlanLayout::Rows {
            row_width,
            start,
            count,
        } => Ok(EdgeLayout::Rows {
            row_width: u64::try_from(*row_width).map_err(|_| Error::ResourceLimit)?,
            start: u64::try_from(*start).map_err(|_| Error::ResourceLimit)?,
            count: u64::try_from(*count).map_err(|_| Error::ResourceLimit)?,
        }),
        PlanLayout::Columns {
            row_width,
            start,
            count,
        } => Ok(EdgeLayout::Columns {
            row_width: u64::try_from(*row_width).map_err(|_| Error::ResourceLimit)?,
            start: u64::try_from(*start).map_err(|_| Error::ResourceLimit)?,
            count: u64::try_from(*count).map_err(|_| Error::ResourceLimit)?,
        }),
        PlanLayout::Transpose { rows, cols } => Ok(EdgeLayout::Transpose {
            rows: u64::try_from(*rows).map_err(|_| Error::ResourceLimit)?,
            cols: u64::try_from(*cols).map_err(|_| Error::ResourceLimit)?,
        }),
        PlanLayout::Broadcast { rows } => Ok(EdgeLayout::Broadcast {
            rows: u64::try_from(*rows).map_err(|_| Error::ResourceLimit)?,
        }),
        PlanLayout::Chain(parts) => {
            let mut converted = Vec::with_capacity(parts.len());
            for part in parts {
                converted.push(edge_layout(part)?);
            }
            Ok(EdgeLayout::Chain(converted))
        }
    }
}

fn source_spec(source: &PlanSource) -> Result<SourceSpec> {
    match source {
        PlanSource::Prev {
            step,
            output,
            layout,
        } => Ok(SourceSpec {
            kind: SourceKind::Prev {
                step: *step,
                output: *output,
            },
            layout: edge_layout(layout)?,
        }),
        PlanSource::Asset { path, layout } => Ok(SourceSpec {
            kind: SourceKind::Asset { path: path.clone() },
            layout: edge_layout(layout)?,
        }),
        PlanSource::Layout(inner, layout) => {
            let mut spec = source_spec(inner)?;
            spec.layout = EdgeLayout::Chain(vec![spec.layout, edge_layout(layout)?]);
            Ok(spec)
        }
        _ => Err(Error::UnsupportedPrimitive),
    }
}

fn asset_meta<'a>(
    public: &'a ModelCommitmentPublic,
    path: &str,
) -> Result<&'a AssetMeta> {
    public
        .assets
        .iter()
        .find(|asset| asset.path == path)
        .ok_or(Error::AssetMismatch)
}

/// Builds the typed edge from the verifier's independent plan. A previous-step
/// source produces a `Prev` edge with no asset identity; an asset source
/// produces an `Asset` edge with the asset identity. Mixed fields are not
/// accepted.
pub fn edge_from_plan(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    producer_step: usize,
    consumer_step: usize,
    input_port: usize,
) -> Result<FoldeqEdgeBinding> {
    let consumer = plan.steps.get(consumer_step).ok_or(Error::InvalidIndex)?;
    let source = consumer.inputs.get(input_port).ok_or(Error::InvalidIndex)?;
    let spec = source_spec(source)?;
    match spec.kind {
        SourceKind::Prev { step, output } => {
            if step != producer_step {
                return Err(Error::StatementMismatch);
            }
            let producer = plan.steps.get(step).ok_or(Error::InvalidIndex)?;
            if output >= producer.outputs {
                return Err(Error::ShapeMismatch);
            }
            Ok(FoldeqEdgeBinding::Prev(PrevEdge {
                producer_step: producer_step as u64,
                producer_output: output as u64,
                consumer_step: consumer_step as u64,
                input_port: input_port as u64,
                direction: EdgeDirection::Forward,
                layout: spec.layout,
            }))
        }
        SourceKind::Asset { path } => {
            let meta = asset_meta(public, &path)?;
            Ok(FoldeqEdgeBinding::Asset(AssetEdge {
                consumer_step: consumer_step as u64,
                input_port: input_port as u64,
                direction: EdgeDirection::Forward,
                asset_id: meta.asset_id,
                path: path.into_bytes(),
                start: meta.start,
                root: meta.root,
                rows: meta.rows,
                row_width: meta.row_width as u64,
                element_count: meta.element_count,
                layout: spec.layout,
            }))
        }
    }
}

/// Rebuilds the plan edge and requires exact equality with a supplied typed
/// edge. This is the independent-plan identity check; it rejects a missing
/// producer, wrong port/layout/slice, self-edge, or wrong source kind.
pub fn check_edge_against_plan(
    public: &ModelCommitmentPublic,
    plan: &TracePlan,
    edge: &FoldeqEdgeBinding,
) -> Result<()> {
    edge.check()?;
    let actual = match edge {
        FoldeqEdgeBinding::NotAnEdge => return Err(Error::InvalidConfig),
        FoldeqEdgeBinding::Prev(edge) => edge_from_plan(
            public,
            plan,
            usize::try_from(edge.producer_step).map_err(|_| Error::ResourceLimit)?,
            usize::try_from(edge.consumer_step).map_err(|_| Error::ResourceLimit)?,
            usize::try_from(edge.input_port).map_err(|_| Error::ResourceLimit)?,
        )?,
        FoldeqEdgeBinding::Asset(edge) => edge_from_plan(
            public,
            plan,
            0,
            usize::try_from(edge.consumer_step).map_err(|_| Error::ResourceLimit)?,
            usize::try_from(edge.input_port).map_err(|_| Error::ResourceLimit)?,
        )?,
    };
    if actual != *edge {
        return Err(Error::StatementMismatch);
    }
    Ok(())
}

fn literal_config(plan: &crate::proof::plan::PlanStep) -> Result<Vec<i64>> {
    let mut config = Vec::with_capacity(plan.config.len());
    for value in &plan.config {
        match value {
            PlanConfigValue::Literal(value) => config.push(*value),
            _ => return Err(Error::InvalidConfig),
        }
    }
    Ok(config)
}

fn check_step<'a>(
    plan: &'a TracePlan,
    trace: &'a SharedTrace,
    index: usize,
    opcode: OpCode,
) -> Result<(Vec<i64>, &'a crate::proof::plan::PlanStep, &'a SharedTraceStep)> {
    if plan.steps.len() != trace.steps.len() {
        return Err(Error::StatementMismatch);
    }
    let planned = plan.steps.get(index).ok_or(Error::InvalidIndex)?;
    let observed = trace.steps.get(index).ok_or(Error::InvalidIndex)?;
    if planned.opcode != opcode
        || observed.message.opcode != opcode as u16
        || planned.outputs != 1
        || observed.message.roots.is_empty()
        || observed.external.len() != planned.inputs.len()
        || observed.sources.len() != planned.inputs.len()
    {
        return Err(Error::StatementMismatch);
    }
    let config = literal_config(planned)?;
    if observed.message.config != config {
        return Err(Error::StatementMismatch);
    }
    Ok((config, planned, observed))
}

fn collect_asset_values(opening: &AssetOpening, count: usize) -> Result<Vec<i64>> {
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

fn authenticate_asset(
    public: &ModelCommitmentPublic,
    path: &str,
    opening: &AssetOpening,
    count: usize,
) -> Result<Vec<i64>> {
    let meta = asset_meta(public, path)?;
    if opening.asset_id != meta.asset_id
        || opening.start != meta.start
        || opening.root != meta.root
        || opening.row_count != meta.rows
        || opening.rows.len() != meta.rows as usize
    {
        return Err(Error::StatementMismatch);
    }
    public.verify_opening(opening)?;
    collect_asset_values(opening, count)
}

fn registry_from_roots(roots: &[(Digest, Digest, u64)]) -> Result<OracleRegistry> {
    let mut registry = OracleRegistry::new();
    for (root, start, count) in roots {
        registry.insert(*root, *start, *count)?;
    }
    Ok(registry)
}

fn relation_public(
    observed: &SharedTraceStep,
    opcode: OpCode,
    config: Vec<i64>,
    registry: OracleRegistry,
) -> Result<OpRelationPublic> {
    let proof = &observed.message.proof;
    let input_oracles: Vec<OracleId> = proof
        .inputs
        .iter()
        .map(|group| {
            group
                .first()
                .map(|opening| opening.oracle)
                .unwrap_or(OracleId(u64::MAX))
        })
        .collect();
    let output_oracles: Vec<OracleId> = proof
        .outputs
        .iter()
        .map(|group| {
            group
                .first()
                .map(|opening| opening.oracle)
                .ok_or(Error::InvalidProof)
        })
        .collect::<Result<_>>()?;
    Ok(OpRelationPublic {
        opcode,
        config,
        registry,
        input_oracles,
        output_oracles,
    })
}

fn decode_group(openings: &[OracleOpening], expected: usize) -> Result<Vec<i64>> {
    let mut values = Vec::with_capacity(expected);
    for opening in openings {
        for value in &opening.values {
            values.push(*value as i64);
        }
    }
    if values.len() != expected {
        return Err(Error::ShapeMismatch);
    }
    Ok(values)
}

/// Executes the real `ProfileLayerNorm` relation check for the selected step
/// and returns its authenticated output. Inputs supplied by an asset opening
/// are authenticated separately; input openings are checked by the relation.
pub fn verify_profile_norm_step(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    plan: &TracePlan,
    step_index: usize,
) -> Result<ProducerRelation> {
    let (config, planned, observed) = check_step(plan, trace, step_index, OpCode::ProfileLayerNorm)?;
    if planned.inputs.len() != 4 || observed.message.proof.inputs.len() != 4 {
        return Err(Error::ShapeMismatch);
    }
    let mut external_values: Vec<Option<Vec<i64>>> = vec![None; 4];
    let mut asset_roots = Vec::new();
    for (port, source) in planned.inputs.iter().enumerate() {
        let spec = source_spec(source)?;
        match spec.kind {
            SourceKind::Prev { step, output } => {
                if observed.external[port].is_some()
                    || observed.sources[port] != Some((step, output))
                {
                    return Err(Error::StatementMismatch);
                }
            }
            SourceKind::Asset { path } => {
                if let Some(pool) = observed.external[port] {
                    let opening = trace.assets.get(pool).ok_or(Error::StatementMismatch)?;
                    let count = *observed
                        .message
                        .proof
                        .lengths
                        .inputs
                        .get(port)
                        .ok_or(Error::ShapeMismatch)?;
                    let values = authenticate_asset(public, &path, opening, count)?;
                    asset_roots.push((port, opening.root));
                    external_values[port] = Some(values);
                } else {
                    // The plan names an asset source; its relation opening is
                    // checked by the relation. This does not claim the opening
                    // is the asset commitment; that distinction remains a
                    // boundary.
                    let _ = asset_meta(public, &path)?;
                    if observed
                        .message
                        .proof
                        .inputs
                        .get(port)
                        .map_or(true, Vec::is_empty)
                    {
                        return Err(Error::ShapeMismatch);
                    }
                }
            }
        }
    }
    let registry = registry_from_roots(&observed.message.roots)?;
    let relation = relation_public(observed, OpCode::ProfileLayerNorm, config, registry)?;
    verify_op_relation_with_external(
        &relation,
        &observed.message.proof,
        &relation.config,
        &external_values,
    )?;
    let output_openings = observed
        .message
        .proof
        .outputs
        .first()
        .ok_or(Error::InvalidProof)?;
    let output = decode_group(
        output_openings,
        *observed
            .message
            .proof
            .lengths
            .outputs
            .first()
            .ok_or(Error::ShapeMismatch)?,
    )?;
    let output_oracle = output_openings
        .first()
        .map(|opening| opening.oracle)
        .ok_or(Error::InvalidProof)?;
    let output_root = relation
        .registry
        .get(output_oracle)
        .ok_or(Error::StatementMismatch)?
        .root;
    Ok(ProducerRelation {
        step_index,
        output,
        output_root,
        output_oracle,
        asset_roots,
    })
}

fn authenticate_group(
    registry: &OracleRegistry,
    openings: &[OracleOpening],
    expected: usize,
) -> Result<(Vec<i64>, Digest, OracleId)> {
    if openings.is_empty() || expected == 0 {
        return Err(Error::ShapeMismatch);
    }
    let count = openings.len();
    if !count.is_power_of_two() || expected % count != 0 {
        return Err(Error::ShapeMismatch);
    }
    let width = expected / count;
    let depth = count.trailing_zeros() as usize;
    let oracle = openings[0].oracle;
    for (position, opening) in openings.iter().enumerate() {
        if opening.oracle != oracle
            || opening.index != position as u64
            || opening.values.len() != width
            || opening.path.len() != depth
        {
            return Err(Error::ShapeMismatch);
        }
        verify_opening(registry, opening, count as u64)?;
    }
    let root = registry.get(oracle).ok_or(Error::StatementMismatch)?.root;
    Ok((decode_group(openings, expected)?, root, oracle))
}

/// Authenticates the consumer's input-0 group against its trace roots. This is
/// the "necessary opening authentication" for the edge; it does not run the
/// consumer relation.
pub fn authenticate_consumer_input(
    plan: &TracePlan,
    trace: &SharedTrace,
    consumer_step: usize,
    input_port: usize,
) -> Result<ConsumerInput> {
    let (_, planned, observed) = check_step(plan, trace, consumer_step, OpCode::MatMul)?;
    let source = planned.inputs.get(input_port).ok_or(Error::InvalidIndex)?;
    let SourceSpec {
        kind: SourceKind::Prev { step, output },
        ..
    } = source_spec(source)?
    else {
        return Err(Error::UnsupportedPrimitive);
    };
    if observed.sources.get(input_port).copied().flatten() != Some((step, output))
        || observed.external.get(input_port).copied().flatten().is_some()
    {
        return Err(Error::StatementMismatch);
    }
    let registry = registry_from_roots(&observed.message.roots)?;
    let group = observed
        .message
        .proof
        .inputs
        .get(input_port)
        .ok_or(Error::InvalidIndex)?;
    let expected = *observed
        .message
        .proof
        .lengths
        .inputs
        .get(input_port)
        .ok_or(Error::ShapeMismatch)?;
    let (values, root, oracle) = authenticate_group(&registry, group, expected)?;
    Ok(ConsumerInput {
        step_index: consumer_step,
        port: input_port,
        values,
        root,
        oracle,
    })
}

/// Compares the two authenticated values. This is the local edge comparison
/// layer; it is not a relation check or a commitment equality claim.
pub fn compare_edge_values(producer: &[i64], consumer: &[i64]) -> Result<()> {
    if producer.len() != consumer.len() {
        return Err(Error::ShapeMismatch);
    }
    if producer != consumer {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Runs the two-step edge demonstration and binds the typed edge into the
/// candidate column-session first query. The upstream input of step `4443` is
/// not verified; the producer's own input remains a boundary.
pub fn verify_edge_session(
    public: &ModelCommitmentPublic,
    trace: &SharedTrace,
    plan: &TracePlan,
    trusted: &TrustedModel<'_>,
    binding: &ColumnBindingInput,
    asset_path: &str,
    opening: &AssetOpening,
) -> Result<EdgeSessionOutcome> {
    verify_model_anchor(
        public,
        trusted.anchor_id,
        trusted.model_id,
        trusted.parameters_digest,
        trusted.asset_specs,
    )?;
    let edge = edge_from_plan(public, plan, 4443, 4444, 0)?;
    check_edge_against_plan(public, plan, &edge)?;
    let producer = verify_profile_norm_step(public, trace, plan, 4443)?;
    let consumer = authenticate_consumer_input(plan, trace, 4444, 0)?;
    compare_edge_values(&producer.output, &consumer.values)?;
    let edge_input = binding.clone().with_edge(edge.clone());
    let column_session = verify_column_session(public, trusted, &edge_input, asset_path, opening)?;
    if !column_session.edge_bound {
        return Err(Error::InvalidProof);
    }
    Ok(EdgeSessionOutcome {
        edge,
        producer_step: 4443,
        consumer_step: 4444,
        input_port: 0,
        producer_output: producer.output,
        consumer_input: consumer.values,
        producer_root: producer.output_root,
        consumer_root: consumer.root,
        relation_verified: true,
        input_authenticated: true,
        values_equal: true,
        column_session,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_value_comparison_is_exact_and_shaped() {
        assert!(compare_edge_values(&[1, 2, 3], &[1, 2, 3]).is_ok());
        assert!(compare_edge_values(&[1, 2, 3], &[1, 2, 4]).is_err());
        assert!(compare_edge_values(&[1, 2], &[1, 2, 3]).is_err());
    }

    #[test]
    fn previous_and_asset_edges_have_distinct_semantics() {
        let prev = FoldeqEdgeBinding::Prev(PrevEdge {
            producer_step: 1,
            producer_output: 0,
            consumer_step: 2,
            input_port: 0,
            direction: EdgeDirection::Forward,
            layout: EdgeLayout::Flat,
        });
        assert!(prev.check().is_ok());
        let asset = FoldeqEdgeBinding::Asset(AssetEdge {
            consumer_step: 2,
            input_port: 0,
            direction: EdgeDirection::Forward,
            asset_id: 1,
            path: b"asset".to_vec(),
            start: Digest([1; 32]),
            root: Digest([2; 32]),
            rows: 1,
            row_width: 1,
            element_count: 1,
            layout: EdgeLayout::Flat,
        });
        assert!(asset.check().is_ok());
        assert_ne!(prev, asset);
        assert!(FoldeqEdgeBinding::Prev(PrevEdge {
            producer_step: 2,
            producer_output: 0,
            consumer_step: 2,
            input_port: 0,
            direction: EdgeDirection::Forward,
            layout: EdgeLayout::Flat,
        })
        .check()
        .is_err());
        assert!(FoldeqEdgeBinding::Asset(AssetEdge {
            consumer_step: 2,
            input_port: 0,
            direction: EdgeDirection::Forward,
            asset_id: 1,
            path: Vec::new(),
            start: Digest([1; 32]),
            root: Digest([2; 32]),
            rows: 1,
            row_width: 1,
            element_count: 1,
            layout: EdgeLayout::Flat,
        })
        .check()
        .is_err());
    }
}
