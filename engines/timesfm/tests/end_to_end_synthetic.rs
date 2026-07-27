use std::path::Path;

use rayon::prelude::*;
use zkie_compiler::dag::{build_dag, link, Commitment, MockProver, Prover};
use zkie_ie_timesfm::{fixtures, partition};

fn fixture_partition_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthetic_partition.toml"
    ))
}

#[test]
fn synthetic_timesfm_shaped_program_links_successfully() {
    let program = fixtures::synthetic_program();
    let specs = partition::load_partition_file(fixture_partition_path()).expect("fixture parses");
    let dag = build_dag(&program, &specs).expect("valid partition");

    assert_eq!(dag.shards.len(), 5);
    // 4 sequential hand-offs (prologue->layer0->layer1->layer2->epilogue)
    // + 3 mask broadcast edges (prologue->each layer) + 1 denorm-stats
    // broadcast edge (prologue->epilogue) = 8 edges.
    assert_eq!(dag.edges.len(), 8);

    let witness = fixtures::synthetic_witness();
    // Every shard's proof is independent of every other's -- prove them
    // all in parallel, demonstrating the whole point of sharding.
    let proofs: Vec<_> = dag
        .shards
        .par_iter()
        .map(|shard| MockProver.prove(shard, &witness))
        .collect();

    assert_eq!(link(&dag, &proofs), Ok(()));
}

#[test]
fn corrupting_one_layer_shard_output_is_caught_by_link() {
    let program = fixtures::synthetic_program();
    let specs = partition::load_partition_file(fixture_partition_path()).expect("fixture parses");
    let dag = build_dag(&program, &specs).expect("valid partition");
    let witness = fixtures::synthetic_witness();

    let mut proofs: Vec<_> = dag
        .shards
        .iter()
        .map(|shard| MockProver.prove(shard, &witness))
        .collect();

    // layer_1 is shard id 2 (prologue=0, layer_0=1, layer_1=2, layer_2=3,
    // epilogue=4); corrupt its one declared output commitment.
    let layer1_output = dag.shards[2].outputs[0].clone();
    proofs[2]
        .output_commitments
        .insert(layer1_output, Commitment([0xFF; 32]));

    assert!(link(&dag, &proofs).is_err());
}
