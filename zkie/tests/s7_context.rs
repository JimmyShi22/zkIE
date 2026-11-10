use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use zkie::provider::StatementData;
use zkie::s7_context::{find_lm_head_matmul, load_gpt2_lm_head_context};
use zkie_core::types::Digest;
use zkie_ops::proof::model_commitment::ModelCommitmentPublic;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn artifacts() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let root = repo_root();
    let case = root.join("models/gpt2/case.toml");
    let policy = root.join("models/gpt2/case.policy");
    let base = root.join(
        "docs/benchmarks/20261110-0547-verify-bytes-109934bd-670a98c9/prod/gpt2-seq16",
    );
    let public = base.join("commit/public");
    let statement = base.join("proof/statement.bin");
    if case.exists()
        && policy.exists()
        && public.join("model-commitment.bin").exists()
        && statement.exists()
    {
        Some((case, policy, public, statement))
    } else {
        None
    }
}

fn read_statement(path: &Path) -> StatementData {
    bincode::deserialize(&fs::read(path).expect("statement bytes")).expect("statement")
}

fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("zkie-s7-{}-{name}", std::process::id()));
    fs::write(&path, bytes).expect("write temp file");
    path
}

#[test]
fn the_compiled_plan_has_one_lm_head_matmul_step() {
    let plan = zkie_models_gpt2::plan::trace_plan(16).expect("plan");
    let step = find_lm_head_matmul(&plan).expect("lm_head matmul");
    assert_eq!(step.opcode, zkie_ops::op::OpCode::MatMul as u16);
    assert_eq!(step.config, vec![16, 65536, 1024, 0, 0]);
    assert_eq!(step.weight_port, 1);
    assert_eq!(step.activation_port, 0);
    assert_eq!(step.output_index, 0);
    assert!(step.activation_step < step.index);
    assert_eq!(step.weight_path, "gpt2/weights/lm_head_w_i32.bin");
}

#[test]
fn the_real_context_is_loaded_with_the_lm_head_identity() {
    let Some((case, policy, public, statement)) = artifacts() else {
        return;
    };
    let statement_data = read_statement(&statement);
    let context = load_gpt2_lm_head_context(
        &case,
        &policy,
        &public,
        &statement,
        statement_data.model_commitment,
    )
    .expect("trusted context");
    assert_eq!(context.asset.path, "gpt2/weights/lm_head_w_i32.bin");
    assert_eq!(context.asset.row_width, 65_536);
    assert_eq!(context.asset.element_count, 1024 * 65_536);
    assert_eq!(context.asset.rows, 1024);
    assert_ne!(context.asset.table_identity, Digest::ZERO);
    assert_ne!(context.asset.start, Digest::ZERO);
    assert_ne!(context.asset.root, Digest::ZERO);
    assert_eq!(context.matmul.config, vec![16, 65_536, 1024, 0, 0]);
    assert!(context.plan_steps > 0);
    println!(
        "S7A asset_id={} table_identity={:?} start={:?} root={:?} leaf_root={:?} rows={} row_width={} elements={} matmul_step={} activation_step={} weight_port={} config={:?} anchor_id={:?} model_id={:?}",
        context.asset.asset_id,
        context.asset.table_identity,
        context.asset.start,
        context.asset.root,
        context.asset.leaf_root,
        context.asset.rows,
        context.asset.row_width,
        context.asset.element_count,
        context.matmul.index,
        context.matmul.activation_step,
        context.matmul.weight_port,
        context.matmul.config,
        context.anchor_id,
        context.model_id
    );
    let weight_path = context.asset_root.join(&context.asset.path);
    if weight_path.exists() {
        let row = context.read_lm_head_row(0).expect("row read");
        assert_eq!(row.len(), 65_536);
        let mut file = fs::File::open(&weight_path).expect("weight file");
        let mut head = [0u8; 32];
        file.read_exact(&mut head).expect("weight head");
        for (index, chunk) in head.chunks_exact(4).enumerate() {
            let raw = i32::from_le_bytes(chunk.try_into().expect("i32"));
            assert_eq!(row[index], raw as i64);
        }
        assert!(context.read_lm_head_row(1024).is_err());
    }
}

#[test]
fn wrong_identities_and_shapes_are_rejected() {
    let Some((case, policy, public, statement)) = artifacts() else {
        return;
    };
    let statement_data = read_statement(&statement);
    let model_id = statement_data.model_commitment;

    assert!(load_gpt2_lm_head_context(&case, &policy, &public, &statement, Digest::ZERO).is_err());

    let policy_text = fs::read_to_string(&policy).expect("policy text");
    let wrong_policy = policy_text.replace(
        "dd08df8bcbba41adea6182709e00adb2f897fde4abc241d1ba09c552fd77ae36",
        &"00".repeat(32),
    );
    let wrong_policy = temp_file("policy.toml", wrong_policy.as_bytes());
    assert!(
        load_gpt2_lm_head_context(&case, &wrong_policy, &public, &statement, model_id).is_err()
    );

    let mut wrong_statement = statement_data.clone();
    wrong_statement.model_commitment = Digest::ZERO;
    let wrong_statement = temp_file(
        "statement-id.bin",
        &bincode::serialize(&wrong_statement).expect("statement"),
    );
    assert!(
        load_gpt2_lm_head_context(&case, &policy, &public, &wrong_statement, model_id).is_err()
    );

    let mut short_statement = statement_data.clone();
    short_statement.tokens.pop();
    let short_statement = temp_file(
        "statement-short.bin",
        &bincode::serialize(&short_statement).expect("statement"),
    );
    assert!(
        load_gpt2_lm_head_context(&case, &policy, &public, &short_statement, model_id).is_err()
    );

    let commitment_bytes = fs::read(public.join("model-commitment.bin")).expect("commitment");
    let mut wrong_public: ModelCommitmentPublic =
        bincode::deserialize(&commitment_bytes).expect("public bundle");
    wrong_public.anchor.model_commitment = Digest::ZERO;
    let anchor_dir = std::env::temp_dir().join(format!("zkie-s7-anchor-{}", std::process::id()));
    fs::create_dir_all(&anchor_dir).expect("temp anchor dir");
    fs::write(
        anchor_dir.join("model-commitment.bin"),
        bincode::serialize(&wrong_public).expect("public"),
    )
    .expect("write public");
    assert!(load_gpt2_lm_head_context(&case, &policy, &anchor_dir, &statement, model_id).is_err());

    let mut wrong_asset: ModelCommitmentPublic =
        bincode::deserialize(&commitment_bytes).expect("public bundle");
    let lm_head = wrong_asset
        .assets
        .iter_mut()
        .find(|asset| asset.path == "gpt2/weights/lm_head_w_i32.bin")
        .expect("lm_head asset");
    lm_head.element_count += 1;
    let wrong_asset_dir = std::env::temp_dir().join(format!("zkie-s7-assets-{}", std::process::id()));
    fs::create_dir_all(wrong_asset_dir.join("public")).expect("temp public dir");
    fs::write(
        wrong_asset_dir.join("public/model-commitment.bin"),
        bincode::serialize(&wrong_asset).expect("public"),
    )
    .expect("write public");
    assert!(load_gpt2_lm_head_context(
        &case,
        &policy,
        &wrong_asset_dir.join("public"),
        &statement,
        model_id
    )
    .is_err());
}
