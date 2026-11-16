//! Registration check for the derived key plan: the plan is derived from the
//! public configuration, and an explicitly registered candidate table instance
//! must match the trusted meta and the plan's table geometry. The old
//! instance's physical root is different and is never overwritten.

use std::fs;
use std::path::Path;

use zkie_models_gpt2::plan::{
    asset_specs, asset_table_params, derive_table_plan, table_model_identity, TABLE_PLAN_CANDIDATE,
};
use zkie_ops::proof::asset_table::{verify_asset_table_meta, AssetTableMeta};
use zkie_ops::proof::hidden_send_session::{production_key_first_plan, KeyPlanInputs};
use zkie_ops::proof::table_range::production_table_range;

fn planned_table() -> (
    zkie_models_gpt2::plan::DerivedTablePlan,
    zkie_ops::proof::hidden_send_session::KeyFirstPlan,
) {
    let derived = derive_table_plan(&TABLE_PLAN_CANDIDATE).unwrap();
    let matrix_rows = derived.table_shape.rows();
    let k_lo = TABLE_PLAN_CANDIDATE.k_lo;
    let columns = matrix_rows << k_lo;
    let inputs = KeyPlanInputs {
        table_identity: derived.table_identity,
        table_configuration: derived.table_configuration,
        table_shape: derived.table_shape,
        matrix_rows,
        k_lo,
        columns,
        total_columns: derived.one_time_columns,
        lin_check: derived.lin_check,
        range: production_table_range().unwrap(),
    };
    let plan = production_key_first_plan(&inputs).unwrap();
    (derived, plan)
}

#[test]
fn the_key_plan_derives_from_the_public_configuration() {
    let (derived, plan) = planned_table();
    let summary = plan.summary();
    eprintln!("key-plan summary: {:?}", summary);
    assert_eq!(summary.table_identity, derived.table_identity);
    assert_eq!(summary.table_configuration, derived.table_configuration);
    assert_eq!(summary.matrix_rows, derived.table_shape.rows());
    assert_eq!(summary.columns, derived.table_shape.rows() << TABLE_PLAN_CANDIDATE.k_lo);
    assert_eq!(summary.index_rows, 1);
    assert_eq!(summary.shared_stages, 3);
    assert_eq!(summary.own_stages, 3);
    assert_eq!(summary.shared_base_positions, 66);
    assert_eq!(summary.own_base_positions, 53);
    assert_eq!(summary.column_budget, derived.one_time_columns as usize);
    // The instance's total column budget covers the one-time test and the
    // declared later batch; the key stage's own budget is the one-time part.
    assert_eq!(derived.one_time_columns, 52);
    assert_eq!(derived.later_batch_columns, 52);
    assert_eq!(derived.declared_later_batches, 1);
    assert_eq!(derived.total_columns, 104);
    assert_eq!(summary.lookup_bits_oracle, 21);
    assert_eq!(summary.lookup_mask_oracle, 22);
    assert_eq!(summary.group_fresh, 3);
    assert_eq!(summary.group_mask_tree, 204);
}

#[test]
fn the_registered_candidate_instance_matches_the_plan() {
    let Some(instance) = std::env::var_os("ZKIE_KEY_INSTANCE") else {
        return;
    };
    let instance = Path::new(&instance);
    let derived = derive_table_plan(&TABLE_PLAN_CANDIDATE).unwrap();
    let metas: Vec<AssetTableMeta> = bincode::deserialize(
        &fs::read(instance.join("commit/public/asset-tables.bin")).unwrap(),
    )
    .unwrap();
    let mut params = asset_table_params().unwrap();
    assert_eq!(params.len(), 1);
    let params = params.pop().unwrap();
    let spec = asset_specs()
        .into_iter()
        .find(|spec| spec.path == params.path)
        .unwrap();
    let meta = metas
        .iter()
        .find(|meta| meta.path == params.path)
        .expect("the registered candidate has the table");
    verify_asset_table_meta(meta, table_model_identity().unwrap(), &spec, &params).unwrap();
    assert_eq!(meta.identity, derived.table_identity);
    assert_eq!(meta.configuration, derived.table_configuration);
    assert_eq!(meta.shape, derived.table_shape);
    // The plan binds the registered instance's reader identity, which covers
    // the committed root and record beside the configuration and shape.
    let reader = zkie_engine::pcs::request::table::IndexReader {
        shape: meta.shape,
        configuration: meta.configuration,
        root: meta.root,
        record: meta.record.clone(),
    };
    let reader_identity =
        zkie_engine::pcs::request::foldeq::reader_table_identity(&reader).unwrap();
    let matrix_rows = derived.table_shape.rows();
    let k_lo = TABLE_PLAN_CANDIDATE.k_lo;
    let plan = production_key_first_plan(&KeyPlanInputs {
        table_identity: reader_identity,
        table_configuration: derived.table_configuration,
        table_shape: derived.table_shape,
        matrix_rows,
        k_lo,
        columns: matrix_rows << k_lo,
        total_columns: derived.one_time_columns,
        lin_check: derived.lin_check,
        range: production_table_range().unwrap(),
    })
    .unwrap();
    let summary = plan.summary();
    assert_eq!(summary.table_identity, reader_identity);
    assert_eq!(meta.configuration, summary.table_configuration);
    assert_ne!(meta.root, zkie_core::types::Digest::ZERO);
    eprintln!(
        "key-plan instance: root={:?} record_len={} identity={:?} reader_identity={:?} configuration={:?} plan={:?}",
        meta.root,
        meta.record.y.len(),
        meta.identity,
        reader_identity,
        meta.configuration,
        summary.plan_digest
    );
    if let Some(old) = std::env::var_os("ZKIE_KEY_OLD_INSTANCE") {
        let old = Path::new(&old);
        let old_metas: Vec<AssetTableMeta> = bincode::deserialize(
            &fs::read(old.join("commit/public/asset-tables.bin")).unwrap(),
        )
        .unwrap();
        let old_meta = old_metas
            .iter()
            .find(|meta| meta.path == params.path)
            .expect("the old instance has the table");
        assert_ne!(old_meta.root, meta.root);
        eprintln!(
            "old instance root={:?} differs from the candidate root={:?}",
            old_meta.root, meta.root
        );
    }
}
