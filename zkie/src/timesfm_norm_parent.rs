//! TimesFM adaptation of the shared norm segment: the first per-layer RMS
//! norm (L0 attention) and its gain-only parameter table.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::field::E_DEGREE;
use zkie_engine::pcs::request::runtime_tree::{RuntimeTreePart, RuntimeTreePlan};
use zkie_ops::proof::asset_table::{AssetTableMeta, AssetTableParams};
use zkie_ops::proof::hidden_norm_psc::NormWitness;
use zkie_ops::proof::model_commitment::SharedTrace;
use zkie_ops::proof::norm_param_table::{
    NormParamClaimPlan, NormParamKind, NormParamTablesPlan, NormParamTableIdentity,
    NORM_PARAM_TABLE_VERSION,
};
use zkie_ops::proof::norm_segment::{
    attach_lookup, attach_lookup_witness, attach_output, attach_output_witness, attach_params,
    attach_params_witness, attach_rstd, attach_rstd_witness, build_core_plan, build_core_witness,
    NormSegmentProfile,
};

const GAIN_PATH: &str = "models/timesfm/weights/L0_lnw_i32.bin";
const SALT: usize = 40;

/// The patch count the norm-only segment is defined for. The segment proves
/// the L0 attention pre-norm of the sixteen patch tokens; the model input is
/// the 512 raw points, so the segment positions come from the case patch
/// count, never from the raw token list.
pub const SEGMENT_POSITIONS: usize = 16;

fn param_claim(kind: NormParamKind, shape: zkie_engine::pcs::request::table::IndexShape) -> Result<NormParamClaimPlan> {
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    let code = build_code_plan(17, 65_536, 8_192)?;
    Ok(NormParamClaimPlan {
        kind,
        gu: GuShape {
            oracle: 0x8800 + kind.encode(),
            group: 0x42,
            tree: RuntimeTreePlan {
                tree_number: 0x1234,
                leaf_class: 1,
                log_n: code.log_n,
                log_n_zk: code.log_n,
                parts: vec![RuntimeTreePart {
                    main_symbols: 2 * E_DEGREE as u64,
                    mask_symbols: 0,
                }],
                salt_bytes: SALT,
                run: Digest([0x71; 32]),
                configuration: Digest([0x72; 32]),
            },
            part: 0,
            rows_high: true,
            u: GuOracleShape {
                code,
                main_rows: 2,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: shape,
        },
        column_draws: 2,
    })
}

/// The gain-only parameter identity of the L0 norm.
pub fn param_identities(metas: &[AssetTableMeta]) -> Result<NormParamTablesPlan> {
    use zkie_ops::proof::model_commitment::asset_table_identity;
    let spec = zkie_models_timesfm::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == GAIN_PATH)
        .ok_or(Error::InvalidConfig)?;
    let meta = metas
        .iter()
        .find(|meta| meta.path == GAIN_PATH)
        .ok_or(Error::MissingCommitArtifact)?;
    let identity = NormParamTableIdentity {
        version: NORM_PARAM_TABLE_VERSION,
        kind: NormParamKind::Gain,
        model_identity: zkie_models_timesfm::plan::table_model_identity()?,
        asset_table_identity: asset_table_identity(&spec),
        reader: zkie_engine::pcs::request::table::IndexReader {
            shape: meta.shape,
            configuration: meta.configuration,
            root: meta.root,
            record: meta.record.clone(),
        },
        row: 0,
        width: zkie_models_timesfm::norm_profile::HIDDEN_PAD as u64,
        valid_min: -i64::from(zkie_models_timesfm::norm_profile::GAIN_MAX),
        valid_max: i64::from(zkie_models_timesfm::norm_profile::GAIN_MAX),
        trusted_i32_mapping: true,
    };
    let gain_claim = param_claim(NormParamKind::Gain, identity.reader.shape)?;
    Ok(NormParamTablesPlan {
        gain: identity,
        bias: None,
        column_draws: 2,
        seed_budget: 1,
        gain_claim: Some(gain_claim),
        bias_claim: None,
    })
}

/// The first per-layer RMS norm step's materialized input and output rows.
pub fn first_rms_norm_rows(
    trace: &SharedTrace,
) -> Result<(usize, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    let target = zkie_ops::op::OpCode::ProfileRmsNorm as u16;
    let (index, step) = trace
        .steps
        .iter()
        .enumerate()
        .find(|(_, step)| step.message.opcode == target)
        .ok_or(Error::MissingCommitArtifact)?;
    let config = &step.message.config;
    if config.len() < 3 {
        return Err(Error::InvalidConfig);
    }
    let rows = usize::try_from(config[0]).map_err(|_| Error::InvalidConfig)?;
    let cols = usize::try_from(config[1]).map_err(|_| Error::InvalidConfig)?;
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidConfig);
    }
    let values = |opening: &zkie_ops::proof::binding::OracleOpening| {
        opening.values.iter().map(|value| *value as i64).collect::<Vec<_>>()
    };
    let x = if let Some(Some((producer, port))) = step.sources.first().copied() {
        trace
            .steps
            .get(producer)
            .and_then(|step| step.message.proof.outputs.get(port))
            .and_then(|outputs| outputs.first())
            .map(values)
            .ok_or(Error::MissingCommitArtifact)?
    } else {
        step.message
            .proof
            .inputs
            .first()
            .and_then(|inputs| inputs.first())
            .map(values)
            .ok_or(Error::MissingCommitArtifact)?
    };
    let hd = step
        .message
        .proof
        .outputs
        .first()
        .and_then(|outputs| outputs.first())
        .map(values)
        .ok_or(Error::MissingCommitArtifact)?;
    if x.len() != rows * cols || hd.len() != rows * cols {
        return Err(Error::ShapeMismatch);
    }
    let split = |flat: &[i64]| {
        (0..rows)
            .map(|row| flat[row * cols..(row + 1) * cols].to_vec())
            .collect::<Vec<_>>()
    };
    Ok((index, split(&x), split(&hd)))
}

/// The L0 gain vector, read through the anchored asset record.
pub fn read_gain(
    asset_root: &std::path::Path,
    assets: &[zkie_ops::proof::model_commitment::AssetMeta],
) -> Result<Vec<i64>> {
    let spec = zkie_models_timesfm::plan::asset_specs()
        .into_iter()
        .find(|spec| spec.path == GAIN_PATH)
        .ok_or(Error::InvalidConfig)?;
    let meta = assets
        .iter()
        .find(|asset| asset.path == GAIN_PATH)
        .ok_or(Error::AssetMismatch)?;
    let (values, root) =
        zkie_ops::proof::model_commitment::read_asset_values_with_root(asset_root, &spec)?;
    if root != meta.leaf_record.leaf_root
        || meta.element_count != spec.element_count
        || meta.row_width != spec.row_width
    {
        return Err(Error::StatementMismatch);
    }
    Ok(values)
}

/// Builds the L0 RMS norm plan at the production shape.
pub fn build_norm_plan(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
) -> Result<zkie_ops::proof::hidden_norm_psc::NormPlan> {
    let space = crate::norm_parent::norm_segment_space(configuration);
    let mut plan = build_core_plan(&space, profile, x_rows)?;
    attach_params(&mut plan, param_identities(metas)?)?;
    attach_rstd(
        &mut plan,
        &space,
        &crate::norm_parent::norm_rstd_space(),
        zkie_models_timesfm::norm_profile::rstd_table_config(),
        zkie_engine::pcs::request::field::Fp3::from_u64(65_536_000),
        zkie_models_timesfm::norm_profile::RSTD_VALUE_BOUND,
    )?;
    attach_output(
        &mut plan,
        profile,
        &space,
        &crate::norm_parent::norm_output_groups(),
    )?;
    attach_lookup(&mut plan, &space, &crate::norm_parent::norm_lookup_space())?;
    Ok(plan)
}

/// Builds the private witness of the L0 norm plan.
pub fn build_norm_witness(
    plan: &zkie_ops::proof::hidden_norm_psc::NormPlan,
    profile: &NormSegmentProfile,
    configuration: Digest,
    x_rows: &[Vec<i64>],
    gain: &[i64],
    rstd: &[i64],
) -> Result<NormWitness> {
    let space = crate::norm_parent::norm_segment_space(configuration);
    let mut witness = build_core_witness(plan, profile, &space, x_rows, rstd)?;
    attach_params_witness(&mut witness, gain, None)?;
    attach_output_witness(&mut witness, plan, profile, gain, None, rstd)?;
    attach_lookup_witness(&mut witness, plan)?;
    attach_rstd_witness(&mut witness, plan)?;
    Ok(witness)
}

/// Builds the L0 norm plan and its private witness.
pub fn build_norm_segment(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
    gain: &[i64],
    rstd: &[i64],
) -> Result<(zkie_ops::proof::hidden_norm_psc::NormPlan, NormWitness)> {
    let plan = build_norm_plan(profile, configuration, metas, x_rows)?;
    let witness = build_norm_witness(&plan, profile, configuration, x_rows, gain, rstd)?;
    Ok((plan, witness))
}

/// The plan-fixed public seed of the L0 norm's x group.
pub fn segment_x_seed(model: Digest, positions: usize) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/timesfm-norm-segment/x/v1");
    bytes.extend_from_slice(&model.0);
    bytes.extend_from_slice(&(positions as u64).to_le_bytes());
    zkie_core::artifact::digest_bytes(&bytes)
}

/// The plan-fixed session seed, bound to the norm-only declaration.
pub fn segment_session_seed(model: Digest, positions: usize, norm_only: bool) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/timesfm-norm-segment/session/v1");
    bytes.extend_from_slice(&model.0);
    bytes.extend_from_slice(&(positions as u64).to_le_bytes());
    bytes.push(u8::from(norm_only));
    zkie_core::artifact::digest_bytes(&bytes)
}

/// The timesfm norm parameter table inputs are built by the shared commit
/// sampler; this helper returns the trusted parameters for the L0 gain table.
pub fn param_table_params(params: &[AssetTableParams]) -> Option<&AssetTableParams> {
    params.iter().find(|params| params.path == GAIN_PATH)
}

use zkie_engine::pcs::request::foldeq::OmegaShapePlan;
use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
use zkie_engine::pcs::request::runtime_tree::MaskTreePlan;
use zkie_engine::pcs::request::table::{IndexShape, TableDistances};
use zkie_ops::proof::hidden_send_session::{
    ActivationShape, ActivationSource, HiddenLinPlan, HiddenOutputShape, HiddenProximityPlan,
    HiddenSendPlan, ReadLedgerCell, WShape,
};

fn segment_gu_shape(code: zkie_engine::pcs::request::code::CodeParams, index: IndexShape) -> GuShape {
    let tree = RuntimeTreePlan {
        tree_number: 10,
        leaf_class: 0,
        log_n: code.log_n,
        log_n_zk: code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: 2 * E_DEGREE as u64,
            mask_symbols: 0,
        }],
        salt_bytes: 40,
        run: Digest([0x41; 32]),
        configuration: Digest([0x42; 32]),
    };
    GuShape {
        oracle: 10,
        group: 7,
        tree,
        part: 0,
        rows_high: true,
        u: GuOracleShape {
            code,
            main_rows: 2,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        },
        index,
    }
}

fn segment_omega_shape(code: zkie_engine::pcs::request::code::CodeParams) -> OmegaShapePlan {
    let tree = RuntimeTreePlan {
        tree_number: 11,
        leaf_class: 1,
        log_n: code.log_n,
        log_n_zk: code.log_n,
        parts: vec![RuntimeTreePart {
            main_symbols: E_DEGREE as u64,
            mask_symbols: 0,
        }],
        salt_bytes: 40,
        run: Digest([0x43; 32]),
        configuration: Digest([0x44; 32]),
    };
    OmegaShapePlan {
        oracle: 11,
        group: 7,
        tree,
        u: GuOracleShape {
            code,
            main_rows: 1,
            main_deg: E_DEGREE as u32,
            mask_rows: 0,
            mask_code: None,
            answers: 1,
        },
        rows_high: true,
        matrix_rows: u64::from(zkie_models_timesfm::norm_profile::HIDDEN_PAD),
        k_lo: 0,
    }
}

/// Builds the complete norm-only plan for the L0 norm. The norm members and
/// the own-stage derivation come from the shared session; the activation root
/// is the materialized hd row identity.
pub fn build_norm_only_plan(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
    hd_rows: &[Vec<i64>],
) -> Result<HiddenLinPlan> {
    let norm_plan = build_norm_plan(profile, configuration, metas, x_rows)?;
    let code = build_code_plan(17, 65_536, 8_192)?;
    let mask_code = build_code_plan(17, 512, 256)?;
    let table_shape = IndexShape {
        code,
        v: 11,
        answers: 1,
        salt_bytes: 40,
        distances: TableDistances {
            radius: 0,
            later_radius: 0,
            m_once: 0,
            m_later: 0,
        },
    };
    let positions = profile.positions;
    let mask_tree = |tree_number: u64| MaskTreePlan {
        tree_number,
        leaf_class: 3,
        log_n: mask_code.log_n,
        mask_rows: 1,
        salt_bytes: 40,
        run: Digest([0x45; 32]),
        configuration,
    };
    let w = WShape::new(
        1,
        code,
        1,
        positions,
        40,
        0,
        0,
        Digest([0x46; 32]),
        configuration,
    )?;
    let proximity = HiddenProximityPlan {
        gu: segment_gu_shape(code, table_shape),
        omega: segment_omega_shape(code),
        code,
        mask_code,
        shared_mask_tree: mask_tree(0),
        own_mask_tree: mask_tree(1),
        // The fresh-mask oracle ids stay below the norm group oracle bases
        // (9000+) and above every member's BAT registration id. The norm-only
        // member set registers more than ninety BAT oracles, so the earlier
        // small ids collided with a member; these values follow the accepted
        // final-norm configuration.
        shared_fresh: 50,
        own_fresh: 1_000,
        column_draws: 1,
        column_budget: 1,
        ledger: ReadLedgerCell::new(1),
        column_work_bits: 0,
        shared_base_positions: 1,
        own_base_positions: 0,
        shared_stages: Vec::new(),
        own_stages: Vec::new(),
        planner_checked: false,
        lin_check: None,
    };
    let send = HiddenSendPlan {
        activation: ActivationShape::new(positions, zkie_models_timesfm::norm_profile::HIDDEN_PAD, 40)?,
        activation_configuration: configuration,
        activation_root: Digest([0x47; 32]),
        activation_source: ActivationSource::NormHdNoDot,
        table_shape,
        table_identity: Digest([0x48; 32]),
        matrix_rows: u64::from(zkie_models_timesfm::norm_profile::HIDDEN_PAD),
        k_lo: 0,
        columns: u64::from(zkie_models_timesfm::norm_profile::HIDDEN_PAD),
        group: 7,
        oracle_base: 20,
        openings_per_oracle: 2,
        w,
        index_mask: None,
        dot: None,
        expected_dot_root: None,
        expected_roots: None,
        member: None,
        request_chain: None,
        chain_mask: None,
        request_reads: None,
        request_w: None,
        source_data: None,
        segment_mask_tree: None,
        range: None,
        digit_lookups: Vec::new(),
    };
    let mut plan = HiddenLinPlan {
        send,
        output: HiddenOutputShape {
            positions,
            columns: u64::from(zkie_models_timesfm::norm_profile::HIDDEN_PAD),
        },
        weights: Vec::new(),
        masks: vec![Vec::new(); positions as usize],
        proximity,
        output_relation: None,
        closure: None,
        norm: Some(norm_plan),
    };
    let roots = zkie_ops::proof::norm_segment::out_group_roots(
        plan.norm.as_ref().ok_or(Error::InvalidConfig)?,
        profile,
        hd_rows,
    )?;
    plan.send.activation_root = zkie_ops::proof::hidden_send_session::hd_activation_root(&plan, &roots)?;
    let own = zkie_ops::proof::hidden_send_session::norm_only_own_stages(&plan)?;
    plan.proximity.own_stages = own.stages;
    plan.proximity.own_base_positions = own.base_positions;
    plan.proximity.own_mask_tree.mask_rows = u64::try_from(own.mask_rows)
        .map_err(|_| Error::ResourceLimit)?;
    plan.validate()?;
    Ok(plan)
}

/// Reads the frozen rstd value of every position by index from the shared
/// GPT-2 rsqrt table under the timesfm asset root.
pub fn read_rstd_values(
    asset_root: &std::path::Path,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<Vec<i64>> {
    use std::io::{Read, Seek, SeekFrom};
    let path = asset_root.join("models/gpt2/weights/rsqrt_table_i32.bin");
    let mut file = std::fs::File::open(path).map_err(|_| Error::AssetMismatch)?;
    let mut values = Vec::with_capacity(x_rows.len());
    for row in x_rows {
        let core = zkie_ops::proof::hidden_norm::NormCoreRow::from_values_mode(row, profile.valid as usize, profile.mode)?;
        let index = zkie_ops::registry::profile_rsqrt_index(core.variance)?;
        if index >= profile.table_real_len {
            return Err(Error::BoundViolation);
        }
        file.seek(SeekFrom::Start(index * 4)).map_err(|_| Error::Io)?;
        let mut entry = [0u8; 4];
        file.read_exact(&mut entry).map_err(|_| Error::Io)?;
        values.push(i32::from_le_bytes(entry) as i64);
    }
    Ok(values)
}

/// The authenticated bounds checks behind the segment's trusted values.
pub fn check_authenticated_bounds(
    plan: &zkie_ops::proof::hidden_norm_psc::NormPlan,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
    gain: &[i64],
    table: &zkie_ops::proof::lookup_table::RangeTable<
        zkie_ops::proof::lookup_table::PublicFileTable,
    >,
) -> Result<()> {
    let round = plan.round.as_ref().ok_or(Error::InvalidConfig)?;
    let params = plan.params.as_ref().ok_or(Error::InvalidConfig)?;
    let rstd = plan.rstd.as_ref().ok_or(Error::InvalidConfig)?;
    zkie_ops::proof::lookup::LookupTable::check_content(table)?;
    if gain.len() != profile.columns as usize || x_rows.len() != profile.positions as usize {
        return Err(Error::ShapeMismatch);
    }
    for value in gain {
        if *value < params.gain.valid_min || *value > params.gain.valid_max {
            return Err(Error::BoundViolation);
        }
    }
    let bound = i64::try_from(round.input_bound).map_err(|_| Error::ResourceLimit)?;
    for row in x_rows {
        if row.len() != profile.hidden as usize {
            return Err(Error::ShapeMismatch);
        }
        if row.iter().any(|value| *value < -bound || *value > bound) {
            return Err(Error::BoundViolation);
        }
        let core = zkie_ops::proof::hidden_norm::NormCoreRow::from_values_mode(row, profile.valid as usize, profile.mode)?;
        let index = zkie_ops::registry::profile_rsqrt_index(core.variance)?;
        let value = table
            .row_value(usize::try_from(index).map_err(|_| Error::ResourceLimit)?)?;
        if value < 0 || value as u64 > rstd.value_bound {
            return Err(Error::BoundViolation);
        }
    }
    Ok(())
}
