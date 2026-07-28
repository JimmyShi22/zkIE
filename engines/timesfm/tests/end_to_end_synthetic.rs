use std::path::Path;

use rayon::prelude::*;
use zkie_compiler::dag::{build_dag, link, Commitment, EdgeKind, LinkError, MockProver, Prover};
use zkie_compiler::graph_compiler::Register;
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

    let sequential_edges: Vec<_> = dag
        .edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Sequential)
        .collect();
    assert_eq!(sequential_edges.len(), 4);

    // The mask (Virtual(1)) is broadcast from the prologue (shard 0) to
    // every one of the 3 layer shards (1, 2, 3).
    let mask_edges: Vec<_> = dag
        .edges
        .iter()
        .filter(|e| e.register == Register::Virtual(1))
        .collect();
    assert_eq!(mask_edges.len(), 3);
    assert!(mask_edges.iter().all(|e| e.kind == EdgeKind::Broadcast));
    assert!(mask_edges.iter().all(|e| e.producer == 0));
    let mask_consumers: std::collections::BTreeSet<usize> =
        mask_edges.iter().map(|e| e.consumer).collect();
    assert_eq!(mask_consumers, std::collections::BTreeSet::from([1, 2, 3]));

    // denorm_stats (Virtual(2)) is produced by the prologue (shard 0) and
    // consumed only by the epilogue (shard 4), skipping every layer.
    let denorm_edges: Vec<_> = dag
        .edges
        .iter()
        .filter(|e| e.register == Register::Virtual(2))
        .collect();
    assert_eq!(denorm_edges.len(), 1);
    assert_eq!(denorm_edges[0].kind, EdgeKind::Broadcast);
    assert_eq!(denorm_edges[0].producer, 0);
    assert_eq!(denorm_edges[0].consumer, 4);

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
        .insert(layer1_output.clone(), Commitment([0xFF; 32]));

    // layer_1 (shard 2) outputs Virtual(22), which is consumed by layer_2 (shard 3).
    // Corrupting layer_1's output should cause link to fail with:
    // CommitmentMismatch { producer: 2, consumer: 3, register: Virtual(22) }
    let expected_error = LinkError::CommitmentMismatch {
        producer: 2,
        consumer: 3,
        register: layer1_output,
    };
    assert_eq!(link(&dag, &proofs), Err(expected_error));
}
