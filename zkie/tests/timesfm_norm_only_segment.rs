//! Component self-test for the timesfm norm-only segment: commit, execute,
//! build the norm plan/witness, prove and verify the carrier. No three-stage
//! CLI run and no key qualification.

use std::path::Path;

use zkie_core::types::Digest;
use zkie_ops::proof::asset_table::sample_asset_table_inputs;
use zkie_ops::proof::model_commitment::ModelCommitmentPublic;

#[test]
#[ignore = "component norm-only segment self-test"]
fn timesfm_norm_only_round_trip() {
    let root = Path::new("/data/zkie-assets");
    let input_bytes = std::fs::read(root.join("cases/timesfm/input_ts_i32.bin")).unwrap();
    let input: Vec<i64> = input_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
        .collect();
    let params = zkie_models_timesfm::plan::asset_table_params().unwrap();
    let tables = sample_asset_table_inputs(&params).unwrap();
    let commitment =
        zkie_models_timesfm::executor::build_commitment(root, &input, &tables).unwrap();
    let trace = zkie_models_timesfm::executor::execute_shared(&commitment, &input).unwrap();
    let public = ModelCommitmentPublic::from_commitment(&commitment);
    let positions = 16usize;
    let model = commitment.anchor.record.model_commitment;
    let profile = zkie_models_timesfm::norm_profile::segment_profile(
        positions as u32,
        zkie::timesfm_norm_parent::segment_x_seed(model, positions),
    )
    .unwrap();
    let (_, x_rows, hd_rows) =
        zkie::timesfm_norm_parent::first_rms_norm_rows(&trace).unwrap();
    let gain = zkie::timesfm_norm_parent::read_gain(root, &public.assets).unwrap();
    let rstd = zkie::timesfm_norm_parent::read_rstd_values(root, &profile, &x_rows).unwrap();
    let configuration = zkie::timesfm_norm_segment_provider::configuration(model);
    let _probe = zkie::timesfm_norm_parent::build_norm_plan(
        &profile,
        configuration,
        &commitment.tables,
        &x_rows,
    )
    .unwrap();
    let plan = zkie::timesfm_norm_parent::build_norm_only_plan(
        &profile,
        configuration,
        &commitment.tables,
        &x_rows,
        &hd_rows,
    )
    .unwrap();
    let norm_plan = plan.norm.clone().unwrap();
    let witness = zkie::timesfm_norm_parent::build_norm_witness(
        &norm_plan,
        &profile,
        configuration,
        &x_rows,
        &gain,
        &rstd,
    )
    .unwrap();
    let table = zkie::timesfm_norm_segment_provider::rstd_table_view(root).unwrap();
    zkie::timesfm_norm_parent::check_authenticated_bounds(
        &norm_plan,
        &profile,
        &x_rows,
        &gain,
        &table,
    )
    .unwrap();
    // R2: compare the rstd key's actual round-message (index, value) pairs
    // with the public table rows; the key sources are read from the committed
    // round rows, not recomputed from an external file.
    {
        let rstd_plan = plan.norm.as_ref().expect("norm plan").rstd.as_ref().expect("rstd plan");
        let round = witness.round.as_ref().expect("round witness");
        let messages = &round.committed.messages;
        for pair in rstd_plan.key.sources.chunks(2) {
            let idx_src = &pair[0];
            let val_src = &pair[1];
            assert_eq!(idx_src.position, val_src.position);
            let position = idx_src.position as usize;
            let idx_value = messages[position][idx_src.row as usize][idx_src.place as usize];
            let val_value = messages[position][val_src.row as usize][val_src.place as usize];
            let core = zkie_ops::proof::hidden_norm::NormCoreRow::from_values_mode(
                &x_rows[position],
                profile.valid as usize,
                profile.mode,
            )
            .unwrap();
            let index = zkie_ops::registry::profile_rsqrt_index(core.variance).unwrap();
            assert_eq!(
                idx_value,
                zkie_engine::pcs::request::field::Fp3::from_u64(index),
                "rstd key index mismatch at position {position}"
            );
            let table_value = table.row_value(index as usize).unwrap();
            assert_eq!(
                val_value,
                zkie_engine::pcs::request::field::Fp3::from_u64(table_value as u64),
                "rstd key value mismatch at position {position}"
            );
        }
        eprintln!("rstd key pairs ok");
    }
    let mut param_tables =
        zkie::timesfm_norm_segment_provider::build_param_tables(&commitment, &tables).unwrap();
    let seed = zkie::timesfm_norm_parent::segment_session_seed(model, positions, true);
    let carrier = zkie::timesfm_norm_segment_provider::prove_bytes(
        &plan,
        seed,
        witness,
        &mut param_tables,
        &table,
    )
    .unwrap();
    let bytes = carrier.encode().unwrap();
    eprintln!(
        "carrier bytes={} first={} second={}",
        bytes.len(),
        carrier.first.len(),
        carrier.second.len()
    );
    let decoded = zkie::timesfm_norm_segment_provider::TimesfmNormCarrier::decode(&bytes).unwrap();
    let checked = zkie::timesfm_norm_segment_provider::verify_bytes(&plan, &decoded, &table);
    assert!(checked.is_ok(), "verify: {checked:?}");
    let _ = Digest::ZERO;
}

/// Prints the own two-stage candidate accounting for the real timesfm plan.
/// This is acceptance evidence for the size sweep; it does not prove.
#[test]
#[ignore]
fn timesfm_norm_only_candidate_bounds() {
    use zkie_engine::pcs::request::code::build_code_plan;
    let root = Path::new("/data/zkie-assets");
    let input_bytes = std::fs::read(root.join("cases/timesfm/input_ts_i32.bin")).unwrap();
    let input: Vec<i64> = input_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
        .collect();
    let params = zkie_models_timesfm::plan::asset_table_params().unwrap();
    let tables = sample_asset_table_inputs(&params).unwrap();
    let commitment =
        zkie_models_timesfm::executor::build_commitment(root, &input, &tables).unwrap();
    let trace = zkie_models_timesfm::executor::execute_shared(&commitment, &input).unwrap();
    let positions = 16usize;
    let model = commitment.anchor.record.model_commitment;
    let profile = zkie_models_timesfm::norm_profile::segment_profile(
        positions as u32,
        zkie::timesfm_norm_parent::segment_x_seed(model, positions),
    )
    .unwrap();
    let (_, x_rows, hd_rows) =
        zkie::timesfm_norm_parent::first_rms_norm_rows(&trace).unwrap();
    let configuration = zkie::timesfm_norm_segment_provider::configuration(model);
    let plan = zkie::timesfm_norm_parent::build_norm_only_plan(
        &profile,
        configuration,
        &commitment.tables,
        &x_rows,
        &hd_rows,
    )
    .unwrap();
    let code = plan.proximity.code;
    let mask_code = plan.proximity.mask_code;
    let own = zkie_ops::proof::hidden_send_session::norm_only_own_stages(&plan).unwrap();
    eprintln!(
        "current: positions={} work={} k={} h={} base={} mask_rows={}",
        own.stages[0].positions,
        own.stages[0].work_bits,
        own.stages[0].k,
        own.stages[0].h,
        own.base_positions,
        own.mask_rows
    );
    let last_main = build_code_plan(code.log_n, code.ell, 1).unwrap();
    let last_mask = build_code_plan(mask_code.log_n, 16_384, 1).unwrap();
    eprintln!(
        "round_group: columns={} main_rows={} ell={} x_rows={} x_ell={}",
        plan.norm.as_ref().unwrap().round.as_ref().unwrap().group.columns,
        plan.norm.as_ref().unwrap().round.as_ref().unwrap().group.shape.main_rows,
        plan.norm.as_ref().unwrap().round.as_ref().unwrap().group.shape.code.ell,
        plan.norm.as_ref().unwrap().x.shape.main_rows,
        plan.norm.as_ref().unwrap().x.shape.code.ell,
    );
    for cap in [252usize, 192, 160, 128] {
        let report = match zkie_ops::proof::proximity_planner::plan_own_two_stage_report(
            code,
            mask_code,
            last_main,
            last_mask,
            own.stages[0].k,
            own.stages[0].h,
            cap,
        ) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("cap={cap} rejected: {error:?}");
                continue;
            }
        };
        eprintln!(
            "cap={cap} stage0_positions={} stage0_work={} stage0_m={} stage0_query_bits={:.6} stage0_carried_bits={:.6} stage0_target_bits={:.6} stage0_joint_bits={:.6} last_positions={} last_r={} last_m={} last_base_bits={:.6} last_target_bits={:.6}",
            report.plan.stage_zero_positions,
            report.plan.stage_zero_work_bits,
            report.plan.stage_zero_m,
            report.stage_zero_query_bits,
            report.stage_zero_carried_bits,
            report.stage_zero_target_bits,
            report.stage_zero_joint_bits,
            report.plan.last_positions,
            report.plan.last_main_r,
            report.plan.last_m,
            report.last_base_bits,
            report.last_target_bits,
        );
    }
}
