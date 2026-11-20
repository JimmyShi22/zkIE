//! Production assembly of the gpt2 final norm segment inside the accepted
//! hidden output-head session.
//!
//! The session keeps the accepted output-head geometry (the table/shape
//! constants of the model output head), adds the norm plan from the segment
//! builder and selects the norm hidden-output object as the MatMul
//! activation. The activation identity is derived from the materialized
//! MatMul activation rows with the plan-fixed public coins, so the prover
//! cannot choose a different object.

use zkie_core::error::{Error, Result};
use zkie_core::types::Digest;
use zkie_engine::pcs::request::code::build_code_plan;
use zkie_engine::pcs::request::field::{Fp3, E_DEGREE};
use zkie_engine::pcs::request::runtime_tree::{
    MaskTreePlan, RuntimeTreePart, RuntimeTreePlan,
};
use zkie_engine::pcs::request::table::{IndexReader, IndexShape};
use zkie_ops::proof::hidden_norm_psc::NormPlan;
use zkie_engine::pcs::request::foldeq::reader_table_identity;
use zkie_ops::proof::hidden_send_session::{
    hd_activation_root, ActivationCommitment, ActivationShape, ActivationSource, DotMapPlan,
    HiddenLinPlan, HiddenProximityPlan, HiddenSendPlan, IndexMaskPlan, IoppStagePlan,
    MaskEntryAllocation, ReadLedgerCell, WShape,
};
use zkie_ops::proof::norm_segment::{out_group_roots, NormSegmentProfile};
use zkie_ops::proof::output_closure::ClosureMaskPlan;
use zkie_ops::proof::output_head_lin::{
    build_output_head_hidden_plan, build_output_head_output_plan,
    build_output_head_output_plan_for, build_output_head_output_plan_unmasked_for,
    output_head_shape_for, OutputClosureSetup, OutputHeadLinSetup, OutputHeadTrustedInputs,
    OUTPUT_HEAD_ASSET_PATH, OUTPUT_HEAD_COLUMNS, OUTPUT_HEAD_HIDDEN, OUTPUT_HEAD_K_LO,
    OUTPUT_HEAD_POSITIONS, OUTPUT_HEAD_RANGE_BITS, OUTPUT_HEAD_SLACK_BITS,
};
use zkie_ops::proof::plan::TracePlan;
use zkie_ops::proof::psc::{hp_entries, psc_variables};

const SALT: usize = 40;
const W_RUN: Digest = Digest([0x41; 32]);
const MASK_RUN: Digest = Digest([0x63; 32]);
const GU_RUN: Digest = Digest([0x32; 32]);
const OMEGA_RUN: Digest = Digest([0x33; 32]);
const GROUP: u64 = 7;
const ORACLE_BASE: u64 = 20;
const GU_ORACLE: u64 = 10;
const OMEGA_ORACLE: u64 = 11;
const PROXIMITY_GROUP: u64 = 5;
const SHARED_FRESH: u64 = 50;
const OWN_FRESH: u64 = 1000;
const W_TREE_NUMBER: u64 = 100;
const SHARED_MASK_TREE: u64 = 200;
const OWN_MASK_TREE: u64 = 201;
const GU_TREE_NUMBER: u64 = 300;
const OMEGA_TREE_NUMBER: u64 = 301;
const INDEX_MASK_ORACLE: u64 = 40;
const INDEX_MASK_TREE: u64 = 202;
const INDEX_MASK_RUN: Digest = Digest([0x66; 32]);
const DOT_ORACLE: u64 = 41;
const DOT_TREE_NUMBER: u64 = 302;
const DOT_RUN: Digest = Digest([0x67; 32]);
const SLACK_ORACLE_BASE: u64 = 60;
const SLACK_TREE_NUMBER: u64 = 400;
const SLACK_RUN: Digest = Digest([0x68; 32]);
const BITS_ORACLE_BASE: u64 = 100;
const BITS_TREE_NUMBER: u64 = 420;
const BITS_RUN: Digest = Digest([0x69; 32]);
const MAP_ORACLE_BASE: u64 = 180;
const MAP_TREE_NUMBER: u64 = 440;
const MAP_RUN: Digest = Digest([0x6B; 32]);
const CLOSURE_MASK_ORACLE: u64 = 140;
const CLOSURE_MASK_TREE: u64 = 500;
const CLOSURE_MASK_RUN: Digest = Digest([0x6A; 32]);
const COLUMN_DRAWS: usize = 603;
const COLUMN_WORK_BITS: u8 = 12;

/// The explicit output-head geometry of one segment. The accepted gpt2
/// segment is the masked form with the 2^16-column table; a model whose
/// logits are argmaxed directly supplies the unmasked form.
#[derive(Clone, Copy, Debug)]
pub struct SegmentHeadGeometry {
    pub positions: usize,
    pub hidden: usize,
    pub columns: usize,
    pub k_lo: u32,
    pub range_bits: u32,
    pub slack_bits: u32,
    pub masked: bool,
    pub asset_path: &'static str,
}

/// The gemma3 segment geometry: the 2^18-column output head argmaxed
/// directly over the full vocabulary, with the logit bound derived from the
/// measured 2^35.1 maximum.
pub const GEMMA3_SEGMENT_HEAD: SegmentHeadGeometry = SegmentHeadGeometry {
    positions: 16,
    hidden: 1024,
    columns: 262_144,
    k_lo: 8,
    // The trusted closure offset must cover the executor's MatMul outputs:
    // the measured per-position absmax is 118_720_934_707 (2^36.79), so
    // 2^38 - 1 keeps a ~2.3x margin while 2^36 - 1 does not cover it.
    range_bits: 38,
    slack_bits: 42,
    masked: false,
    asset_path: "models/gemma3/weights/lm_head_i32.bin",
};

/// The accepted gpt2 segment geometry.
pub const GPT2_SEGMENT_HEAD: SegmentHeadGeometry = SegmentHeadGeometry {
    positions: OUTPUT_HEAD_POSITIONS,
    hidden: OUTPUT_HEAD_HIDDEN,
    columns: OUTPUT_HEAD_COLUMNS,
    k_lo: OUTPUT_HEAD_K_LO,
    range_bits: OUTPUT_HEAD_RANGE_BITS,
    slack_bits: OUTPUT_HEAD_SLACK_BITS,
    masked: true,
    asset_path: OUTPUT_HEAD_ASSET_PATH,
};

/// The session main and mask codes of one segment geometry.
///
/// This is the single derivation point for the hidden session's group code:
/// the shared stage, the norm, lookup and rstd spaces, the parameter claims
/// and the segment codes all take their main code from here. A wider output
/// head therefore keeps its parameter claims in the same BAT group as the
/// main members instead of a second, hardcoded code family.
pub fn segment_session_codes(
    geometry: &SegmentHeadGeometry,
) -> Result<(
    zkie_engine::pcs::request::code::CodeParams,
    zkie_engine::pcs::request::code::CodeParams,
)> {
    let code_log = (geometry.columns as u64 + 8192)
        .checked_next_power_of_two()
        .ok_or(Error::ResourceLimit)?
        .trailing_zeros();
    Ok((
        build_code_plan(code_log, geometry.columns as u64, 8192)?,
        build_code_plan(code_log, 512, 256)?,
    ))
}

fn mask_tree(tree_number: u64, configuration: Digest, code: zkie_engine::pcs::request::code::CodeParams) -> MaskTreePlan {
    MaskTreePlan {
        tree_number,
        leaf_class: 3,
        log_n: code.log_n,
        mask_rows: 1,
        salt_bytes: SALT,
        run: MASK_RUN,
        configuration,
    }
}

/// Builds the accepted hidden output-head session with the final norm plan
/// and the norm hidden-output object selected as the MatMul activation.
///
/// The activation root is derived from the materialized activation rows with
/// the plan-fixed public coins; the same derivation is used by the verifier.
pub fn build_segment_plan(
    trace_plan: &TracePlan,
    expected_argmax: &[u32],
    table_reader: &IndexReader,
    table_shape: IndexShape,
    table_configuration: Digest,
    norm_plan: &NormPlan,
    profile: &NormSegmentProfile,
    hd_rows: &[Vec<i64>],
    seed: Digest,
) -> Result<HiddenLinPlan> {
    build_segment_plan_for(
        &GPT2_SEGMENT_HEAD,
        trace_plan,
        expected_argmax,
        table_reader,
        table_shape,
        table_configuration,
        norm_plan,
        profile,
        hd_rows,
        seed,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_segment_plan_for(
    geometry: &SegmentHeadGeometry,
    trace_plan: &TracePlan,
    expected_argmax: &[u32],
    table_reader: &IndexReader,
    table_shape: IndexShape,
    table_configuration: Digest,
    norm_plan: &NormPlan,
    profile: &NormSegmentProfile,
    hd_rows: &[Vec<i64>],
    seed: Digest,
) -> Result<HiddenLinPlan> {
    if profile.positions as usize != geometry.positions
        || profile.hidden as usize != geometry.hidden
        || profile.columns != geometry.hidden as u64
    {
        return Err(Error::UnsupportedPrimitive);
    }
    let output_plan = if geometry.masked {
        build_output_head_output_plan_for(
            trace_plan,
            geometry.positions,
            geometry.hidden,
            geometry.columns,
            geometry.asset_path,
            geometry.range_bits,
            geometry.slack_bits,
            expected_argmax,
        )?
    } else {
        build_output_head_output_plan_unmasked_for(
            trace_plan,
            geometry.positions,
            geometry.hidden,
            geometry.columns,
            geometry.asset_path,
            geometry.range_bits,
            geometry.slack_bits,
            expected_argmax,
        )?
    };
    let (code, mask_code) = segment_session_codes(geometry)?;
    // The later stage codes share the session log; it comes from the one
    // derivation above instead of a second formula.
    let code_log = code.log_n;
    let tree = |tree_number: u64, run: Digest, main_rows: u64| -> Result<RuntimeTreePlan> {
        Ok(RuntimeTreePlan {
            tree_number,
            leaf_class: 2,
            log_n: code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: main_rows
                    .checked_mul(E_DEGREE as u64)
                    .ok_or(Error::ResourceLimit)?,
                mask_symbols: 0,
            }],
            salt_bytes: SALT,
            run,
            configuration: table_configuration,
        })
    };
    let w = WShape::new(
        geometry.columns as u64,
        code,
        1,
        geometry.positions as u32,
        SALT,
        W_TREE_NUMBER,
        1,
        W_RUN,
        table_configuration,
    )?;
    let main_rows = w.main_rows;
    // The shared FOLD's fold shape is derived from the shared member set:
    // the committed table's GU and Omega members plus the two parameter
    // claim members. The stage parameters must match that derived shape or
    // the chain refuses the stage-0 handoff.
    let shared_main_rows = {
        let mut rows = main_rows.checked_add(1).ok_or(Error::ResourceLimit)?;
        if let Some(params) = norm_plan.params.as_ref() {
            if let Some(claim) = params.gain_claim.as_ref() {
                rows = rows
                    .checked_add(claim.gu.u.main_rows)
                    .ok_or(Error::ResourceLimit)?;
            }
            if let Some(claim) = params.bias_claim.as_ref() {
                rows = rows
                    .checked_add(claim.gu.u.main_rows)
                    .ok_or(Error::ResourceLimit)?;
            }
        }
        rows
    };
    let shared_fold_k = if shared_main_rows <= 1 {
        1
    } else {
        64 - (shared_main_rows - 1).leading_zeros()
    };
    let proximity = HiddenProximityPlan {
        gu: zkie_engine::pcs::request::gu::GuShape {
            oracle: GU_ORACLE,
            group: PROXIMITY_GROUP,
            tree: tree(GU_TREE_NUMBER, GU_RUN, main_rows)?,
            part: 0,
            rows_high: true,
            u: zkie_engine::pcs::request::gu::GuOracleShape {
                code,
                main_rows,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            index: table_shape,
        },
        omega: zkie_engine::pcs::request::foldeq::OmegaShapePlan {
            oracle: OMEGA_ORACLE,
            group: PROXIMITY_GROUP,
            tree: tree(OMEGA_TREE_NUMBER, OMEGA_RUN, 1)?,
            u: zkie_engine::pcs::request::gu::GuOracleShape {
                code,
                main_rows: 1,
                main_deg: E_DEGREE as u32,
                mask_rows: 0,
                mask_code: None,
                answers: 1,
            },
            rows_high: true,
            matrix_rows: geometry.hidden as u64,
            k_lo: geometry.k_lo,
        },
        code,
        mask_code,
        shared_mask_tree: mask_tree(SHARED_MASK_TREE, table_configuration, mask_code),
        own_mask_tree: MaskTreePlan {
            mask_rows: 4,
            ..mask_tree(OWN_MASK_TREE, table_configuration, mask_code)
        },
        shared_fresh: SHARED_FRESH,
        own_fresh: OWN_FRESH,
        column_draws: COLUMN_DRAWS,
        column_budget: COLUMN_DRAWS,
        ledger: ReadLedgerCell::new(COLUMN_DRAWS as u64),
        column_work_bits: COLUMN_WORK_BITS,
        shared_base_positions: 2,
        own_base_positions: 259,
        own_deferred_samples: true,
        shared_stages: vec![
            IoppStagePlan {
                main: code,
                mask: mask_code,
                k: shared_fold_k,
                h: 0,
                positions: 2,
                samples: 1,
                work_bits: 0,
            },
            IoppStagePlan {
                main: build_code_plan(code_log, geometry.columns as u64, 256)?,
                mask: build_code_plan(code_log, 16_384, 256)?,
                k: 0,
                h: 0,
                positions: 2,
                samples: 1,
                work_bits: 0,
            },
        ],
        own_stages: vec![
            IoppStagePlan {
                main: code,
                mask: mask_code,
                k: 14,
                h: 2,
                positions: 252,
                samples: 1,
                work_bits: 24,
            },
            IoppStagePlan {
                main: build_code_plan(code_log, geometry.columns as u64, 259)?,
                mask: build_code_plan(17, 16_384, 259)?,
                k: 0,
                h: 0,
                positions: 259,
                samples: 1,
                work_bits: 0,
            },
        ],
        planner_checked: false,
        lin_check: None,
    };
    let index_mask = IndexMaskPlan {
        oracle: INDEX_MASK_ORACLE,
        group: GROUP,
        mask_code,
        mask_rows: 2,
        answers: 1,
        openings: 2,
        tree: RuntimeTreePlan {
            tree_number: INDEX_MASK_TREE,
            leaf_class: 2,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 2 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: INDEX_MASK_RUN,
            configuration: table_configuration,
        },
        entries: MaskEntryAllocation::new(geometry.positions as u32)?,
    };
    let index_rows = index_mask.tree_rows()?;
    let norm_mask_rows = norm_plan
        .mask
        .as_ref()
        .map(|mask| mask.mask_rows)
        .unwrap_or(0);
    let lookup_mask_rows = norm_plan
        .lookup
        .as_ref()
        .map(|lookup| lookup.commit.mask_rows)
        .unwrap_or(0);
    let rstd_mask_rows = norm_plan
        .rstd
        .as_ref()
        .map(|rstd| rstd.commit.mask_rows)
        .unwrap_or(0);
    let total_mask_rows = norm_mask_rows
        .checked_add(lookup_mask_rows)
        .and_then(|value| value.checked_add(rstd_mask_rows))
        .ok_or(Error::ResourceLimit)?;
    let own_mask_rows = zkie_ops::proof::hidden_norm_psc::norm_own_mask_rows(
        index_rows,
        total_mask_rows,
    )?;
    let own_h = own_mask_rows.trailing_zeros();
    // The own two-stage chain is a derived quantity: the tree's planner
    // fixes the stage-0 position/work pair and the last tail's row count from
    // the real main/mask codes and the fold shape. The fixture constants are
    // not used here.
    let closure_width = 2 * (geometry.range_bits + 1) + geometry.slack_bits;
    let bits_columns = (geometry.columns as u64)
        .checked_mul(closure_width as u64)
        .ok_or(Error::ResourceLimit)?
        .next_power_of_two();
    let boolean_variables = bits_columns.trailing_zeros() as usize;
    let closure_slack = WShape::new(
        geometry.columns as u64,
        code,
        1,
        geometry.positions as u32,
        SALT,
        SLACK_TREE_NUMBER,
        2,
        SLACK_RUN,
        table_configuration,
    )?;
    // The bit and map classes stack every position block into one message
    // each: the row count, the exact fill and the tail-pad cell come from the
    // stacked shape, not from a per-position message.
    let closure_bits = zkie_ops::proof::output_closure::stacked_shape(
        bits_columns,
        geometry.positions as u32,
        code,
        SALT,
        BITS_TREE_NUMBER,
        2,
        BITS_RUN,
        table_configuration,
    )?;
    let closure_variables = psc_variables(boolean_variables)?;
    let closure_mask = ClosureMaskPlan {
        oracle: CLOSURE_MASK_ORACLE,
        group: GROUP,
        mask_code,
        mask_rows: 2,
        answers: 1,
        openings: 2,
        tree: RuntimeTreePlan {
            tree_number: CLOSURE_MASK_TREE,
            leaf_class: 3,
            log_n: mask_code.log_n,
            log_n_zk: mask_code.log_n,
            parts: vec![RuntimeTreePart {
                main_symbols: 0,
                mask_symbols: 2 * E_DEGREE as u64,
            }],
            salt_bytes: SALT,
            run: CLOSURE_MASK_RUN,
            configuration: table_configuration,
        },
        positions: geometry.positions as u32,
        hiding: hp_entries(&closure_variables)?,
        shown: 5,
    };
    let closure_setup = OutputClosureSetup {
        slack: closure_slack,
        slack_oracle_base: SLACK_ORACLE_BASE,
        slack_openings: 2,
        bits: closure_bits,
        bits_oracle_base: BITS_ORACLE_BASE,
        bits_openings: 2,
        map: zkie_ops::proof::output_closure::stacked_shape(
            bits_columns,
            geometry.positions as u32,
            code,
            SALT,
            MAP_TREE_NUMBER,
            2,
            MAP_RUN,
            table_configuration,
        )?,
        map_oracle_base: MAP_ORACLE_BASE,
        map_openings: 2,
        mask: closure_mask,
    };
    let activation_salts: Vec<Vec<u8>> = (0..geometry.positions)
        .map(|position| vec![(position as u8).wrapping_mul(11) ^ 0x3C; SALT])
        .collect();
    let setup = OutputHeadLinSetup {
        asset_path: geometry.asset_path.to_string(),
        send: HiddenSendPlan {
            activation: ActivationShape::new(
                geometry.positions as u32,
                geometry.hidden as u32,
                SALT,
            )?,
            activation_configuration: table_configuration,
            activation_root: Digest([0x70; 32]),
            activation_source: ActivationSource::Commitment,
            table_shape,
            table_identity: reader_table_identity(table_reader)?,
            matrix_rows: geometry.hidden as u64,
            k_lo: geometry.k_lo,
            columns: geometry.columns as u64,
            group: GROUP,
            oracle_base: ORACLE_BASE,
            openings_per_oracle: 2,
            w,
            index_mask: None,
            dot: Some(DotMapPlan {
                oracle: DOT_ORACLE,
                group: GROUP,
                shape: WShape::new(
                    (geometry.positions * geometry.hidden) as u64,
                    code,
                    1,
                    geometry.positions as u32,
                    SALT,
                    DOT_TREE_NUMBER,
                    2,
                    DOT_RUN,
                    table_configuration,
                )?,
                hidden: geometry.hidden as u32,
            }),
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
        },
        proximity,
        index_mask: Some(index_mask),
        activation_configuration: table_configuration,
        activation_salts,
        seed,
        masks: Vec::new(),
        expected_activation_root: None,
        expected_roots: None,
        closure: Some(closure_setup),
    };
    let placeholder = ActivationCommitment {
        root: Digest([0x70; 32]),
        shape: ActivationShape::new(
            geometry.positions as u32,
            geometry.hidden as u32,
            SALT,
        )?,
        configuration: table_configuration,
    };
    let trusted = OutputHeadTrustedInputs {
        activation: placeholder,
        output: output_plan,
    };
    let mut plan = build_output_head_hidden_plan(
        output_head_shape_for(geometry.hidden as u64, geometry.k_lo, table_reader.shape.rows(), geometry.columns as u64)?,
        &setup,
        table_reader,
        &trusted,
    )?;
    plan.norm = Some(norm_plan.clone());
    plan.send.activation_source = ActivationSource::NormHd;
    let roots = out_group_roots(norm_plan, profile, hd_rows)?;
    let root = hd_activation_root(&plan, &roots)?;
    plan.send.activation = ActivationShape::new(
        geometry.positions as u32,
        geometry.hidden as u32,
        SALT,
    )?;
    plan.send.activation_configuration = table_configuration;
    plan.send.activation_root = root;
    let index_rows = match plan.send.index_mask.as_ref() {
        Some(index) => index.tree_rows()?,
        None => 0,
    };
    let norm = plan.norm.as_ref().ok_or(Error::InvalidConfig)?;
    let norm_mask_rows = norm.mask.as_ref().map(|mask| mask.mask_rows).unwrap_or(0);
    let lookup_mask_rows = norm.lookup.as_ref().map(|lookup| lookup.commit.mask_rows).unwrap_or(0);
    let rstd_mask_rows = norm.rstd.as_ref().map(|rstd| rstd.commit.mask_rows).unwrap_or(0);
    let total_mask_rows = norm_mask_rows
        .checked_add(lookup_mask_rows)
        .and_then(|value| value.checked_add(rstd_mask_rows))
        .ok_or(Error::ResourceLimit)?;
    plan.proximity.own_mask_tree.mask_rows = own_mask_rows;
    if let Some(closure) = plan.closure.as_mut() {
        closure.tail_pad = true;
    }
    // The own fold shape is re-derived from the assembled member set: the
    // BAT layout and the FOLD k follow this sum, so the stage plan is built
    // against it instead of a hand-set constant.
    let own_k = zkie_ops::proof::hidden_send_session::own_member_k(&plan)?;
    let own_two_stage = zkie_ops::proof::proximity_planner::plan_own_two_stage(
        code,
        mask_code,
        build_code_plan(code_log, geometry.columns as u64, 1)?,
        build_code_plan(code_log, 16_384, 1)?,
        own_k,
        own_h,
        252,
    )?;
    plan.proximity.own_base_positions = own_two_stage.last_positions;
    plan.proximity.own_stages[0] = IoppStagePlan {
        main: code,
        mask: mask_code,
        k: own_k,
        h: own_h,
        positions: own_two_stage.stage_zero_positions,
        samples: 1,
        work_bits: own_two_stage.stage_zero_work_bits,
    };
    plan.proximity.own_stages[1] = IoppStagePlan {
        main: build_code_plan(code_log, geometry.columns as u64, own_two_stage.last_main_r)?,
        mask: build_code_plan(code_log, 16_384, own_two_stage.last_mask_r)?,
        k: 0,
        h: 0,
        positions: own_two_stage.last_positions,
        samples: 1,
        work_bits: 0,
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_gemma3_output_range_covers_the_executor_logits() {
        // The executor's published MatMul outputs for the real seq16 case
        // reach 118_720_934_707 per position (2^36.79). The trusted closure
        // offset must cover that with margin. The earlier 2^35.09 figure was
        // a different representation, not the MatMul output.
        const MEASURED_ABSMAX: u64 = 118_720_934_707;
        let offset = (1u64 << super::GEMMA3_SEGMENT_HEAD.range_bits) - 1;
        assert_eq!(super::GEMMA3_SEGMENT_HEAD.range_bits, 38);
        assert!(offset > MEASURED_ABSMAX);
        assert!(offset / MEASURED_ABSMAX >= 2);
        assert!((1u64 << 36) - 1 < MEASURED_ABSMAX);
    }

    use super::*;
    use zkie_engine::pcs::request::code::build_code_plan;
    use zkie_engine::pcs::request::field::Fp3;
    use zkie_engine::pcs::request::table::{IndexRecord, TableDistances};
    use zkie_ops::proof::norm_param_table::{
        NormParamClaimPlan, NormParamKind, NormParamTableIdentity, NormParamTablesPlan,
        NORM_PARAM_TABLE_VERSION,
    };
    use zkie_ops::proof::norm_segment::{
        attach_lookup, attach_output, attach_params, attach_rstd, build_core_plan, GroupIds,
        LookupSpace, NormSegmentProfile, NormSegmentSpace, OutputGroups, RstdSpace,
    };

    fn space() -> NormSegmentSpace {
        NormSegmentSpace {
            group: 7,
            code: build_code_plan(17, 65536, 8192).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
            run: Digest([0x41; 32]),
            configuration: Digest([0x83; 32]),
            salt_bytes: 40,
            leaf_class: 1,
            x: GroupIds { oracle: 9000, tree: 60000 },
            sum: GroupIds { oracle: 9100, tree: 60100 },
            squares: GroupIds { oracle: 9200, tree: 60200 },
            mean: GroupIds { oracle: 9300, tree: 60300 },
            variance: GroupIds { oracle: 9400, tree: 60400 },
            low: GroupIds { oracle: 9700, tree: 60700 },
            high: GroupIds { oracle: 9720, tree: 60720 },
            round: GroupIds { oracle: 9600, tree: 60600 },
            mask: GroupIds { oracle: 9500, tree: 60500 },
        }
    }

    fn output_groups() -> OutputGroups {
        OutputGroups {
            broadcast: GroupIds { oracle: 20000, tree: 61000 },
            product: GroupIds { oracle: 20100, tree: 61100 },
            image: GroupIds { oracle: 20200, tree: 61200 },
            out: GroupIds { oracle: 20300, tree: 61300 },
            tail: GroupIds { oracle: 20400, tree: 61400 },
        }
    }

    fn lookup_space() -> LookupSpace {
        LookupSpace {
            bits_oracle: 9900,
            mask_oracle: 9901,
            bits_tree: 60980,
            mask_tree: 60981,
            code: build_code_plan(17, 65536, 8192).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
        }
    }

    fn rstd_space() -> RstdSpace {
        RstdSpace {
            bits_oracle: 9800,
            mask_oracle: 9801,
            bits_tree: 60900,
            mask_tree: 60901,
            code: build_code_plan(17, 65536, 8192).unwrap(),
            mask_code: build_code_plan(17, 512, 256).unwrap(),
        }
    }

    fn param_reader(root: u8, configuration: u8) -> IndexReader {
        let shape = IndexShape {
            code: build_code_plan(12, 2049, 4).unwrap(),
            v: 0,
            answers: 1,
            salt_bytes: 40,
            distances: TableDistances {
                radius: 0,
                later_radius: 0,
                m_once: 0,
                m_later: 0,
            },
        };
        let record = IndexRecord {
            y: vec![Fp3::ONE; shape.rows() as usize * shape.answers as usize],
        };
        IndexReader {
            shape,
            configuration: Digest([configuration; 32]),
            root: Digest([root; 32]),
            record,
        }
    }

    fn param_claim(kind: NormParamKind, shape: IndexShape) -> NormParamClaimPlan {
        use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
        let code = build_code_plan(12, 2049, 4).unwrap();
        NormParamClaimPlan {
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
                    salt_bytes: 40,
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
        }
    }

    fn norm_plan(profile: &NormSegmentProfile) -> NormPlan {
        let x_rows = vec![vec![1i64; profile.hidden as usize]; profile.positions as usize];
        let mut plan = build_core_plan(&space(), profile, &x_rows).unwrap();
        let gain = param_reader(0x82, 0x81);
        let bias = param_reader(0x84, 0x83);
        let params = NormParamTablesPlan {
            gain_claim: Some(param_claim(NormParamKind::Gain, gain.shape)),
            bias_claim: Some(param_claim(NormParamKind::Bias, bias.shape)),
            gain: NormParamTableIdentity {
                version: NORM_PARAM_TABLE_VERSION,
                kind: NormParamKind::Gain,
                model_identity: Digest([0x63; 32]),
                asset_table_identity: Digest([0x64; 32]),
                reader: gain,
                row: 0,
                width: profile.columns,
                valid_min: -1_141_592,
                valid_max: 1_141_592,
                trusted_i32_mapping: true,
            },
            bias: Some(NormParamTableIdentity {
                version: NORM_PARAM_TABLE_VERSION,
                kind: NormParamKind::Bias,
                model_identity: Digest([0x63; 32]),
                asset_table_identity: Digest([0x64; 32]),
                reader: bias,
                row: 0,
                width: profile.columns,
                valid_min: -482_890,
                valid_max: 482_890,
                trusted_i32_mapping: true,
            }),
            column_draws: 2,
            seed_budget: 1,
        };
        attach_params(&mut plan, params).unwrap();
        attach_rstd(
            &mut plan,
            &space(),
            &rstd_space(),
            zkie_models_gpt2::norm_profile::rstd_table_config(),
            Fp3::from_u64(65_536_000),
            zkie_models_gpt2::norm_profile::RSTD_VALUE_BOUND,
        )
        .unwrap();
        attach_output(&mut plan, profile, &space(), &output_groups()).unwrap();
        attach_lookup(&mut plan, &space(), &lookup_space()).unwrap();
        plan
    }

    fn real_parent_plan() -> HiddenLinPlan {
        let profile = zkie_models_gpt2::norm_profile::segment_profile(16, Digest([0x91; 32])).unwrap();
        let plan_norm = norm_plan(&profile);
        let trace_plan = zkie_models_gpt2::plan::trace_plan(16).unwrap();
        let table_plan = zkie_models_gpt2::lm_head_plan::lm_head_plan().unwrap();
        let table_shape = table_plan.table_shape().unwrap();
        let table_reader = IndexReader {
            shape: table_shape,
            configuration: Digest([0x83; 32]),
            root: Digest([0x84; 32]),
            record: IndexRecord {
                y: vec![
                    Fp3::ONE;
                    table_shape.rows() as usize * table_shape.answers as usize
                ],
            },
        };
        let hd_rows = vec![vec![1i64; profile.columns as usize]; profile.positions as usize];
        let argmax = vec![0u32; 16];
        build_segment_plan(
            &trace_plan,
            &argmax,
            &table_reader,
            table_shape,
            table_reader.configuration,
            &plan_norm,
            &profile,
            &hd_rows,
            Digest([0x72; 32]),
        )
        .unwrap()
    }

    #[test]
    fn the_real_parent_plan_is_admitted() {
        let plan = real_parent_plan();
        assert!(plan.norm.is_some());
        assert!(plan.send.activation_source.is_norm_hd());
        plan.validate().unwrap();
    }

    /// The norm may carry zero padding columns: the valid columns are at most
    /// the matrix rows, and the padded columns are bound through the output
    /// tail padding rows.
    #[test]
    fn a_valid_count_above_the_matrix_rows_is_rejected() {
        let mut plan = real_parent_plan();
        plan.validate().unwrap();
        let matrix_rows = plan.send.matrix_rows;
        plan.norm.as_mut().unwrap().shape.valid = (matrix_rows + 1) as u32;
        assert!(plan.validate().is_err());
        plan.norm.as_mut().unwrap().shape.valid = matrix_rows as u32;
        plan.validate().unwrap();
    }

    /// The norm width must cover the matrix rows; a narrower norm leaves the
    /// MatMul with columns no committed row carries.
    #[test]
    fn a_norm_width_below_the_matrix_rows_is_rejected() {
        let mut plan = real_parent_plan();
        plan.validate().unwrap();
        let matrix_rows = plan.send.matrix_rows;
        let columns = plan.norm.as_ref().unwrap().columns;
        plan.norm.as_mut().unwrap().columns = matrix_rows - 1;
        assert!(plan.validate().is_err());
        plan.norm.as_mut().unwrap().columns = columns;
        plan.validate().unwrap();
    }

    /// The session main and mask codes have one derivation point: both
    /// accepted geometries come from the same function and the gemma3 helper
    /// is that derivation, not a second constant.
    #[test]
    fn the_session_codes_come_from_one_derivation() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (main, mask) = super::segment_session_codes(&super::GPT2_SEGMENT_HEAD).unwrap();
        assert_eq!(main, build_code_plan(17, 65536, 8192).unwrap());
        assert_eq!(mask, build_code_plan(17, 512, 256).unwrap());
        let (gemma_main, gemma_mask) =
            super::segment_session_codes(&super::GEMMA3_SEGMENT_HEAD).unwrap();
        assert_eq!(gemma_main, build_code_plan(19, 262_144, 8192).unwrap());
        assert_eq!(gemma_mask, build_code_plan(19, 512, 256).unwrap());
        assert_eq!(
            super::gemma3_segment_codes().unwrap(),
            (gemma_main, gemma_mask)
        );
    }

    /// D (k=13) component check: the merged closure message shape.
    ///
    /// The two closure classes stack their per-position blocks into one
    /// message each. For gemma3 each position block is 128 rows over the
    /// session code's 2^18 places (2^25 entries per position), so the merged
    /// message holds 16 x 2^25 = 2^29 entries. Its code keeps the session
    /// log_n and r and raises the message length by three pads; the last
    /// entry then lands at place 262143, before the padded capacity
    /// (2^29 / 2^18 = 2048 rows exactly). If any other W shape condition
    /// rejects this, the assertion names it.
    /// The norm group identifiers are plan values: every group's oracle
    /// range and every group's tree range must be pairwise disjoint over the
    /// complete space, not only between the limb and the round group. An
    /// overlap silently maps one group's members onto another group's trees
    /// (same shapes, different roots), which is a soundness defect even when
    /// both shapes are identical.
    #[test]
    fn the_norm_group_identifiers_are_pairwise_disjoint() {
        use zkie_engine::pcs::request::code::build_code_plan;
        let (code, mask_code) = super::gemma3_segment_codes().unwrap();
        let space = super::norm_segment_space_coded(Digest([0x21; 32]), code, mask_code);
        let lookup = super::norm_lookup_space_coded(code, mask_code);
        let rstd = super::norm_rstd_space_coded(code, mask_code);
        let outputs = super::norm_output_groups();
        let positions = super::GEMMA3_SEGMENT_HEAD.positions as u64;

        // (name, first, end) with the trusted span of each group.
        let width = |span: u64| -> u64 { span };
        let mut oracles: Vec<(&str, u64, u64)> = vec![
            ("x", space.x.oracle, space.x.oracle + width(positions)),
            ("sum", space.sum.oracle, space.sum.oracle + width(positions)),
            ("squares", space.squares.oracle, space.squares.oracle + width(positions)),
            ("mean", space.mean.oracle, space.mean.oracle + width(positions)),
            ("variance", space.variance.oracle, space.variance.oracle + width(positions)),
            ("low", space.low.oracle, space.low.oracle + width(positions)),
            ("high", space.high.oracle, space.high.oracle + width(positions)),
            ("round", space.round.oracle, space.round.oracle + width(positions)),
            ("mask", space.mask.oracle, space.mask.oracle + 1),
            ("lookup_bits", lookup.bits_oracle, lookup.bits_oracle + 1),
            ("lookup_mask", lookup.mask_oracle, lookup.mask_oracle + 1),
            ("rstd_bits", rstd.bits_oracle, rstd.bits_oracle + 1),
            ("rstd_mask", rstd.mask_oracle, rstd.mask_oracle + 1),
            ("out_broadcast", outputs.broadcast.oracle, outputs.broadcast.oracle + width(positions)),
            ("out_product", outputs.product.oracle, outputs.product.oracle + width(positions)),
            ("out_image", outputs.image.oracle, outputs.image.oracle + width(positions)),
            ("out_out", outputs.out.oracle, outputs.out.oracle + width(positions)),
            ("out_tail", outputs.tail.oracle, outputs.tail.oracle + width(positions)),
        ];
        // The parameter claims and the closure messages share the same space.
        oracles.extend([
            ("param_gain", 0x8800, 0x8801),
            ("param_bias", 0x8801, 0x8802),
            ("closure_slack", super::SLACK_ORACLE_BASE, super::SLACK_ORACLE_BASE + positions),
            ("closure_bits", super::BITS_ORACLE_BASE, super::BITS_ORACLE_BASE + 1),
            ("closure_map", super::MAP_ORACLE_BASE, super::MAP_ORACLE_BASE + 1),
            ("closure_mask", super::CLOSURE_MASK_ORACLE, super::CLOSURE_MASK_ORACLE + 1),
        ]);
        let mut trees: Vec<(&str, u64, u64)> = vec![
            ("x", space.x.tree, space.x.tree + width(positions)),
            ("sum", space.sum.tree, space.sum.tree + width(positions)),
            ("squares", space.squares.tree, space.squares.tree + width(positions)),
            ("mean", space.mean.tree, space.mean.tree + width(positions)),
            ("variance", space.variance.tree, space.variance.tree + width(positions)),
            ("low", space.low.tree, space.low.tree + width(positions)),
            ("high", space.high.tree, space.high.tree + width(positions)),
            ("round", space.round.tree, space.round.tree + width(positions)),
            ("mask", space.mask.tree, space.mask.tree + 1),
            ("lookup_bits", lookup.bits_tree, lookup.bits_tree + 1),
            ("lookup_mask", lookup.mask_tree, lookup.mask_tree + 1),
            ("rstd_bits", rstd.bits_tree, rstd.bits_tree + 1),
            ("rstd_mask", rstd.mask_tree, rstd.mask_tree + 1),
            ("out_broadcast", outputs.broadcast.tree, outputs.broadcast.tree + width(positions)),
            ("out_product", outputs.product.tree, outputs.product.tree + width(positions)),
            ("out_image", outputs.image.tree, outputs.image.tree + width(positions)),
            ("out_out", outputs.out.tree, outputs.out.tree + width(positions)),
            ("out_tail", outputs.tail.tree, outputs.tail.tree + width(positions)),
        ];
        trees.extend([
            ("closure_slack", super::SLACK_TREE_NUMBER, super::SLACK_TREE_NUMBER + positions),
            ("closure_bits", super::BITS_TREE_NUMBER, super::BITS_TREE_NUMBER + 1),
            ("closure_map", super::MAP_TREE_NUMBER, super::MAP_TREE_NUMBER + 1),
            ("closure_mask", super::CLOSURE_MASK_TREE, super::CLOSURE_MASK_TREE + 1),
        ]);
        for (label, ranges) in [("oracle", &oracles), ("tree", &trees)] {
            for first in 0..ranges.len() {
                for second in first + 1..ranges.len() {
                    let (a_name, a_first, a_end) = ranges[first];
                    let (b_name, b_first, b_end) = ranges[second];
                    assert!(
                        a_end <= b_first || b_end <= a_first,
                        "{label} ranges overlap: {a_name} [{a_first},{a_end}) and {b_name} [{b_first},{b_end})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_merged_closure_shape_is_two_thousand_forty_rows() {
        use zkie_engine::pcs::request::code::build_code_plan;
        use zkie_ops::proof::proximity_planner::rows_for_columns;

        let (code, _mask_code) = super::gemma3_segment_codes().unwrap();
        let geometry = &super::GEMMA3_SEGMENT_HEAD;
        let closure_width = 2 * (geometry.range_bits + 1) + geometry.slack_bits;
        let block_columns = (geometry.columns as u64)
            .checked_mul(closure_width as u64)
            .unwrap()
            .next_power_of_two();
        assert_eq!(block_columns, 1u64 << 25);
        let positions = geometry.positions as u32;
        let entries = block_columns * positions as u64;
        assert_eq!(entries, 1u64 << 29);

        // Each position block keeps the session place count 2^18; the merged
        // code adds three pads so the per-row capacity is exactly 2^18.
        let capacity = code.ell;
        assert_eq!(capacity, 1u64 << 18);
        let answers = 3u32;
        let ell = capacity + answers as u64;
        let merged_code = build_code_plan(code.log_n, ell, code.r).unwrap();
        let block_rows = block_columns / capacity;
        assert_eq!(block_rows, 128);
        assert_eq!(block_rows * positions as u64, 2048);
        assert_eq!(rows_for_columns(entries, ell, answers).unwrap(), 2048);

        let configuration = Digest([0x5A; 32]);
        for tree_number in [super::BITS_TREE_NUMBER, super::MAP_TREE_NUMBER] {
            let shape = WShape::new(
                entries,
                merged_code,
                answers,
                1,
                super::SALT,
                tree_number,
                2,
                super::BITS_RUN,
                configuration,
            )
            .expect("the merged closure shape is a message shape");
            assert_eq!(shape.main_rows, 2048);
            assert!(shape.fault(entries, 1).is_none());
            assert!(shape.oracle_shape().fault().is_none());
        }
    }
}

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use zkie_ops::proof::asset_table::AssetTableMeta;
use zkie_ops::proof::hidden_norm::NormCoreRow;
use zkie_ops::proof::model_commitment::SharedTrace;
use zkie_ops::proof::norm_param_table::{
    NormParamClaimPlan, NormParamKind, NormParamTableIdentity, NormParamTablesPlan,
    NORM_PARAM_TABLE_VERSION,
};
use zkie_ops::proof::norm_segment::{
    attach_lookup, attach_lookup_witness, attach_output, attach_output_witness, attach_params,
    attach_params_witness, attach_rstd, attach_rstd_witness, build_core_plan,
    build_core_witness, GroupIds, LookupSpace, NormSegmentSpace,
    OutputGroups, RstdSpace,
};

/// The trusted constants of the production norm segment.
pub fn norm_segment_space(configuration: Digest) -> NormSegmentSpace {
    let (code, mask_code) =
        segment_session_codes(&GPT2_SEGMENT_HEAD).expect("the session codes");
    norm_segment_space_coded(configuration, code, mask_code)
}

/// The segment space with explicit session codes. A wider output head uses
/// the same main and mask codes as its hidden session, so the norm groups
/// share the proximity group.
pub fn norm_segment_space_coded(
    configuration: Digest,
    code: zkie_engine::pcs::request::code::CodeParams,
    mask_code: zkie_engine::pcs::request::code::CodeParams,
) -> NormSegmentSpace {
    NormSegmentSpace {
        group: 7,
        code,
        mask_code,
        run: Digest([0x41; 32]),
        configuration,
        salt_bytes: SALT,
        leaf_class: 1,
        x: GroupIds { oracle: 9000, tree: 60000 },
        sum: GroupIds { oracle: 9100, tree: 60100 },
        squares: GroupIds { oracle: 9200, tree: 60200 },
        mean: GroupIds { oracle: 9300, tree: 60300 },
        variance: GroupIds { oracle: 9400, tree: 60400 },
        low: GroupIds { oracle: 9700, tree: 60700 },
        high: GroupIds { oracle: 9720, tree: 60720 },
        round: GroupIds { oracle: 9600, tree: 60600 },
        mask: GroupIds { oracle: 9500, tree: 60500 },
    }
}

pub fn norm_lookup_space() -> LookupSpace {
    let (code, mask_code) =
        segment_session_codes(&GPT2_SEGMENT_HEAD).expect("the session codes");
    norm_lookup_space_coded(code, mask_code)
}

pub fn norm_lookup_space_coded(
    code: zkie_engine::pcs::request::code::CodeParams,
    mask_code: zkie_engine::pcs::request::code::CodeParams,
) -> LookupSpace {
    LookupSpace {
        bits_oracle: 9900,
        mask_oracle: 9901,
        bits_tree: 60980,
        mask_tree: 60981,
        code,
        mask_code,
    }
}

pub fn norm_rstd_space() -> RstdSpace {
    let (code, mask_code) =
        segment_session_codes(&GPT2_SEGMENT_HEAD).expect("the session codes");
    norm_rstd_space_coded(code, mask_code)
}

pub fn norm_rstd_space_coded(
    code: zkie_engine::pcs::request::code::CodeParams,
    mask_code: zkie_engine::pcs::request::code::CodeParams,
) -> RstdSpace {
    RstdSpace {
        bits_oracle: 9800,
        mask_oracle: 9801,
        bits_tree: 60900,
        mask_tree: 60901,
        code,
        mask_code,
    }
}

pub fn norm_output_groups() -> OutputGroups {
    OutputGroups {
        broadcast: GroupIds { oracle: 20000, tree: 61000 },
        product: GroupIds { oracle: 20100, tree: 61100 },
        image: GroupIds { oracle: 20200, tree: 61200 },
        out: GroupIds { oracle: 20300, tree: 61300 },
        tail: GroupIds { oracle: 20400, tree: 61400 },
    }
}

fn param_claim(
    kind: NormParamKind,
    shape: IndexShape,
    code: zkie_engine::pcs::request::code::CodeParams,
) -> NormParamClaimPlan {
    use zkie_engine::pcs::request::gu::{GuOracleShape, GuShape};
    NormParamClaimPlan {
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
    }
}

/// The trusted parameter-table identities of the final-norm assets. The
/// layer form carries a gain and a bias row; the RMS form carries the gain
/// row alone.
pub fn param_identities_for(
    metas: &[AssetTableMeta],
    specs: &[zkie_ops::proof::model_commitment::ModelAssetSpec],
    model_identity: Digest,
    code: zkie_engine::pcs::request::code::CodeParams,
    width: u64,
    gain: (&str, i64, i64),
    bias: Option<(&str, i64, i64)>,
) -> Result<NormParamTablesPlan> {
    use zkie_ops::proof::model_commitment::asset_table_identity;
    let identity = |kind: NormParamKind,
                    path: &str,
                    valid_min: i64,
                    valid_max: i64|
     -> Result<NormParamTableIdentity> {
        let spec = specs
            .iter()
            .find(|spec| spec.path == path)
            .ok_or(Error::InvalidConfig)?;
        let meta = metas
            .iter()
            .find(|meta| meta.path == path)
            .ok_or(Error::MissingCommitArtifact)?;
        Ok(NormParamTableIdentity {
            version: NORM_PARAM_TABLE_VERSION,
            kind,
            model_identity,
            asset_table_identity: asset_table_identity(spec),
            reader: IndexReader {
                shape: meta.shape,
                configuration: meta.configuration,
                root: meta.root,
                record: meta.record.clone(),
            },
            row: 0,
            width,
            valid_min,
            valid_max,
            trusted_i32_mapping: true,
        })
    };
    let gain_identity = identity(NormParamKind::Gain, gain.0, gain.1, gain.2)?;
    let gain_claim = param_claim(NormParamKind::Gain, gain_identity.reader.shape, code);
    let (bias_identity, bias_claim) = match bias {
        Some((path, valid_min, valid_max)) => {
            let bias_identity = identity(NormParamKind::Bias, path, valid_min, valid_max)?;
            let bias_claim = param_claim(NormParamKind::Bias, bias_identity.reader.shape, code);
            (Some(bias_identity), Some(bias_claim))
        }
        None => (None, None),
    };
    Ok(NormParamTablesPlan {
        gain: gain_identity,
        bias: bias_identity,
        column_draws: 2,
        seed_budget: 1,
        gain_claim: Some(gain_claim),
        bias_claim,
    })
}

/// The accepted gpt2 parameter tables.
pub fn param_identities(metas: &[AssetTableMeta]) -> Result<NormParamTablesPlan> {
    param_identities_for(
        metas,
        &zkie_models_gpt2::plan::asset_specs(),
        zkie_models_gpt2::plan::table_model_identity()?,
        segment_session_codes(&GPT2_SEGMENT_HEAD)?.0,
        1024,
        ("gpt2/weights/ln_f_w_i32.bin", -1_141_592, 1_141_592),
        Some(("gpt2/weights/ln_f_b_i32.bin", -482_890, 482_890)),
    )
}

/// The gemma3 parameter table: the gain-only final norm.
pub fn param_identities_gemma3(metas: &[AssetTableMeta]) -> Result<NormParamTablesPlan> {
    param_identities_for(
        metas,
        &zkie_models_gemma3::plan::asset_specs(),
        zkie_models_gemma3::plan::table_model_identity()?,
        segment_session_codes(&GEMMA3_SEGMENT_HEAD)?.0,
        1024,
        (
            "models/gemma3/weights/final_norm_i32.bin",
            0,
            zkie_models_gemma3::norm_profile::GAIN_MAX,
        ),
        None,
    )
}

/// The final norm step's materialized input rows and output rows, read from
/// the recorded trace. The input rows are the verified previous step's output;
/// the output rows are the same physical row the output head consumes.
pub fn final_norm_rows(
    trace: &SharedTrace,
) -> Result<(usize, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    final_norm_rows_for(trace, zkie_ops::op::OpCode::ProfileLayerNorm as u16)
}

/// The materialized rows of the last step with the given opcode.
pub fn final_norm_rows_for(
    trace: &SharedTrace,
    target: u16,
) -> Result<(usize, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    let (index, step) = trace
        .steps
        .iter()
        .enumerate()
        .rev()
        .find(|(_, step)| step.message.opcode == target)
        .ok_or(Error::MissingCommitArtifact)?;
    let config = &step.message.config;
    if config.len() < 4 {
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

/// Reads the final-norm parameter vectors from the anchored assets. The file
/// values are decoded through the same encoding as the commitment builder and
/// the recomputed row-leaf root must equal the public leaf record, so only
/// authenticated values reach the checks. The bias vector is present only for
/// the layer form.
pub fn read_anchored_params_for(
    asset_root: &std::path::Path,
    assets: &[zkie_ops::proof::model_commitment::AssetMeta],
    specs: &[zkie_ops::proof::model_commitment::ModelAssetSpec],
    gain_path: &str,
    bias_path: Option<&str>,
) -> Result<(Vec<i64>, Option<Vec<i64>>)> {
    let read = |path: &str| -> Result<Vec<i64>> {
        let spec = specs
            .iter()
            .find(|spec| spec.path == path)
            .ok_or(Error::InvalidConfig)?;
        let meta = assets
            .iter()
            .find(|asset| asset.path == path)
            .ok_or(Error::AssetMismatch)?;
        let (values, root) =
            zkie_ops::proof::model_commitment::read_asset_values_with_root(asset_root, spec)?;
        if root != meta.leaf_record.leaf_root
            || meta.element_count != spec.element_count
            || meta.row_width != spec.row_width
        {
            return Err(Error::StatementMismatch);
        }
        Ok(values)
    };
    let gain = read(gain_path)?;
    if gain.len() != 1024 {
        return Err(Error::AssetMismatch);
    }
    let bias = match bias_path {
        Some(path) => {
            let bias = read(path)?;
            if bias.len() != 1024 {
                return Err(Error::AssetMismatch);
            }
            Some(bias)
        }
        None => None,
    };
    Ok((gain, bias))
}

pub fn read_anchored_params(
    asset_root: &std::path::Path,
    assets: &[zkie_ops::proof::model_commitment::AssetMeta],
) -> Result<(Vec<i64>, Vec<i64>)> {
    let (gain, bias) = read_anchored_params_for(
        asset_root,
        assets,
        &zkie_models_gpt2::plan::asset_specs(),
        "gpt2/weights/ln_f_w_i32.bin",
        Some("gpt2/weights/ln_f_b_i32.bin"),
    )?;
    Ok((gain, bias.ok_or(Error::AssetMismatch)?))
}

pub fn read_anchored_params_gemma3(
    asset_root: &std::path::Path,
    assets: &[zkie_ops::proof::model_commitment::AssetMeta],
) -> Result<Vec<i64>> {
    let (gain, bias) = read_anchored_params_for(
        asset_root,
        assets,
        &zkie_models_gemma3::plan::asset_specs(),
        "models/gemma3/weights/final_norm_i32.bin",
        None,
    )?;
    if bias.is_some() {
        return Err(Error::AssetMismatch);
    }
    Ok(gain)
}

/// The authenticated value checks behind the segment's trusted bounds. The
/// bounds are plan values; the checked values come only from authenticated
/// objects: the materialized trace rows, the anchored parameter assets and
/// the frozen public rstd table. Both the prover and the verifier run it.
pub fn check_authenticated_bounds(
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
    gain: &[i64],
    bias: &[i64],
    table: &dyn zkie_ops::proof::lookup::LookupTable,
) -> Result<()> {
    let round = plan.round.as_ref().ok_or(Error::InvalidConfig)?;
    let params = plan.params.as_ref().ok_or(Error::InvalidConfig)?;
    let rstd = plan.rstd.as_ref().ok_or(Error::InvalidConfig)?;
    // The table read must be the frozen content, not merely a file at the
    // expected path.
    zkie_ops::proof::lookup::LookupTable::check_content(table)?;
    if gain.len() != profile.columns as usize
        || (!bias.is_empty() && bias.len() != profile.columns as usize)
        || x_rows.len() != profile.positions as usize
    {
        return Err(Error::ShapeMismatch);
    }
    for value in gain {
        if *value < params.gain.valid_min || *value > params.gain.valid_max {
            return Err(Error::BoundViolation);
        }
    }
    if let Some(bias_identity) = &params.bias {
        for value in bias {
            if *value < bias_identity.valid_min || *value > bias_identity.valid_max {
                return Err(Error::BoundViolation);
            }
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
        let core = authenticated_core(
            profile,
            plan.rms
                .as_ref()
                .map(|rms| (rms.widths, rms.mean_bound)),
            row,
        )?;
        let index = zkie_ops::registry::profile_rsqrt_index(core.variance)?;
        let value = table
            .row_value(usize::try_from(index).map_err(|_| Error::ResourceLimit)?)?;
        if value < 0 || value as u64 > rstd.value_bound {
            return Err(Error::BoundViolation);
        }
        // The RMS form also checks the correlated pre-clamp bound on the
        // authenticated row: every reachable x * rstd * gain cell stays
        // inside the plan's bound.
        if let Some(rms) = &plan.rms {
            for (column, x) in row.iter().take(profile.valid as usize).enumerate() {
                let image = (*x as i128)
                    .checked_mul(value as i128)
                    .and_then(|value| value.checked_mul(gain[column] as i128))
                    .ok_or(Error::BoundViolation)?;
                let magnitude = image.unsigned_abs();
                if magnitude > rms.pre_bound as u128 {
                    return Err(Error::BoundViolation);
                }
            }
        }
    }
    Ok(())
}

/// The exact core of one authenticated row under the profile's relation. The
/// RMS form uses the multi-limb row with the same mean bound the plan checks;
/// the layer form keeps the default centering core.
fn authenticated_core(
    profile: &NormSegmentProfile,
    rms: Option<(zkie_ops::proof::rms_limbs::RmsLimbWidths, u64)>,
    row: &[i64],
) -> Result<NormCoreRow> {
    if profile.mode.is_rms() {
        let (widths, mean_bound) = match rms {
            Some(value) => value,
            None => (
                zkie_ops::proof::rms_limbs::RmsLimbWidths::derive(
                    profile.input_bound,
                    profile.valid,
                )?,
                profile.mean_bound,
            ),
        };
        NormCoreRow::from_values_rms(
            row,
            profile.valid as usize,
            profile.input_bound,
            widths,
            mean_bound,
        )
    } else {
        NormCoreRow::from_values(row, profile.valid as usize)
    }
}

/// Reads the frozen-table rstd value of every position by index.
pub fn read_rstd_values(
    asset_root: &std::path::Path,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<Vec<i64>> {
    let path = asset_root.join("gpt2/weights/rsqrt_table_i32.bin");
    let mut file = File::open(path).map_err(|_| Error::AssetMismatch)?;
    let mut values = Vec::with_capacity(x_rows.len());
    for row in x_rows {
        let core = authenticated_core(profile, None, row)?;
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

/// Reads the combined frozen-table rstd value of every gemma3 position by
/// index: rows below the base length come from the base file, the rest from
/// the extension file. The index comes from the same multi-limb core the
/// plan commits.
pub fn read_rstd_values_gemma3(
    asset_root: &std::path::Path,
    profile: &NormSegmentProfile,
    x_rows: &[Vec<i64>],
) -> Result<Vec<i64>> {
    let base = asset_root.join("models/gpt2/weights/rsqrt_table_i32.bin");
    let extension = asset_root.join("cases/gemma3/rsqrt_ext_i32.bin");
    let mut base_file = File::open(base).map_err(|_| Error::AssetMismatch)?;
    let mut extension_file = File::open(extension).map_err(|_| Error::AssetMismatch)?;
    let mut values = Vec::with_capacity(x_rows.len());
    for row in x_rows {
        let core = authenticated_core(profile, None, row)?;
        let index = zkie_ops::registry::profile_rsqrt_index(core.variance)?;
        if index >= profile.table_real_len {
            return Err(Error::BoundViolation);
        }
        let (file, offset) = if index < zkie_models_gemma3::norm_profile::RSTD_BASE_LEN {
            (&mut base_file, index)
        } else {
            (
                &mut extension_file,
                index - zkie_models_gemma3::norm_profile::RSTD_BASE_LEN,
            )
        };
        file.seek(SeekFrom::Start(offset * 4)).map_err(|_| Error::Io)?;
        let mut entry = [0u8; 4];
        file.read_exact(&mut entry).map_err(|_| Error::Io)?;
        values.push(i32::from_le_bytes(entry) as i64);
    }
    Ok(values)
}

/// The plan-fixed public seed of the norm segment's x group commitment. The
/// prover and the verifier derive it from the model identity and the position
/// count; the prover cannot choose the x roots.
pub fn segment_x_seed(model: Digest, positions: usize) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/norm-segment/x/v1");
    bytes.extend_from_slice(&model.0);
    bytes.extend_from_slice(&(positions as u64).to_le_bytes());
    zkie_core::artifact::digest_bytes(&bytes)
}

/// The plan-fixed public session seed of the output-head hidden session.
pub fn segment_session_seed(model: Digest, positions: usize, identity_binding: Digest) -> Digest {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"zkie/norm-segment/session/v2");
    bytes.extend_from_slice(&model.0);
    bytes.extend_from_slice(&(positions as u64).to_le_bytes());
    bytes.extend_from_slice(&identity_binding.0);
    zkie_core::artifact::digest_bytes(&bytes)
}

/// Builds the norm plan at the production shape. The verifier rebuilds the
/// same plan from the materialized trace rows and the trusted constants.
pub fn build_norm_plan(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
) -> Result<NormPlan> {
    let space = norm_segment_space(configuration);
    let mut plan = build_core_plan(&space, profile, x_rows)?;
    attach_params(&mut plan, param_identities(metas)?)?;
    attach_rstd(
        &mut plan,
        &space,
        &norm_rstd_space(),
        zkie_models_gpt2::norm_profile::rstd_table_config(),
        Fp3::from_u64(65_536_000),
        zkie_models_gpt2::norm_profile::RSTD_VALUE_BOUND,
    )?;
    attach_output(&mut plan, profile, &space, &norm_output_groups())?;
    attach_lookup(&mut plan, &space, &norm_lookup_space())?;
    Ok(plan)
}

/// The session codes of the gemma3 segment geometry, from the single
/// derivation point.
pub fn gemma3_segment_codes() -> Result<(
    zkie_engine::pcs::request::code::CodeParams,
    zkie_engine::pcs::request::code::CodeParams,
)> {
    segment_session_codes(&GEMMA3_SEGMENT_HEAD)
}

/// Builds the gemma3 norm plan at the production shape: the gain-only RMS
/// form, the combined rstd table and the accepted output/lookup attachments.
pub fn build_norm_plan_gemma3(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
) -> Result<NormPlan> {
    let (code, mask_code) = gemma3_segment_codes()?;
    let space = norm_segment_space_coded(configuration, code, mask_code);
    let mut plan = build_core_plan(&space, profile, x_rows)?;
    attach_params(&mut plan, param_identities_gemma3(metas)?)?;
    attach_rstd(
        &mut plan,
        &space,
        &norm_rstd_space_coded(code, mask_code),
        zkie_models_gemma3::norm_profile::rstd_table_config(),
        Fp3::from_u64(65_536_000),
        zkie_models_gemma3::norm_profile::RSTD_VALUE_BOUND,
    )?;
    attach_output(&mut plan, profile, &space, &norm_output_groups())?;
    attach_lookup(&mut plan, &space, &norm_lookup_space_coded(code, mask_code))?;
    Ok(plan)
}

/// Builds the gain-only gemma3 witness. The bias vector is absent.
pub fn build_norm_witness_gemma3(
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    configuration: Digest,
    x_rows: &[Vec<i64>],
    gain: &[i64],
    rstd: &[i64],
) -> Result<zkie_ops::proof::hidden_norm_psc::NormWitness> {
    let (code, mask_code) = gemma3_segment_codes()?;
    let space = norm_segment_space_coded(configuration, code, mask_code);
    let mut witness = build_core_witness(plan, profile, &space, x_rows, rstd)?;
    attach_params_witness(&mut witness, gain, None)?;
    attach_output_witness(&mut witness, plan, profile, gain, None, rstd)?;
    attach_lookup_witness(&mut witness, plan)?;
    attach_rstd_witness(&mut witness, plan)?;
    Ok(witness)
}

/// Builds the private witness of the plan. Prover side only.
pub fn build_norm_witness(
    plan: &NormPlan,
    profile: &NormSegmentProfile,
    configuration: Digest,
    x_rows: &[Vec<i64>],
    gain: &[i64],
    bias: &[i64],
    rstd: &[i64],
) -> Result<zkie_ops::proof::hidden_norm_psc::NormWitness> {
    let space = norm_segment_space(configuration);
    let mut witness = build_core_witness(plan, profile, &space, x_rows, rstd)?;
    attach_params_witness(&mut witness, gain, Some(bias))?;
    attach_output_witness(&mut witness, plan, profile, gain, Some(bias), rstd)?;
    attach_lookup_witness(&mut witness, plan)?;
    attach_rstd_witness(&mut witness, plan)?;
    Ok(witness)
}

/// Builds the norm plan and its private witness at the production shape.
pub fn build_norm_segment(
    profile: &NormSegmentProfile,
    configuration: Digest,
    metas: &[AssetTableMeta],
    x_rows: &[Vec<i64>],
    gain: &[i64],
    bias: &[i64],
    rstd: &[i64],
) -> Result<(NormPlan, zkie_ops::proof::hidden_norm_psc::NormWitness)> {
    let plan = build_norm_plan(profile, configuration, metas, x_rows)?;
    let witness = build_norm_witness(&plan, profile, configuration, x_rows, gain, bias, rstd)?;
    Ok((plan, witness))
}
