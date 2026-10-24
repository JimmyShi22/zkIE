//! End-to-end vertical weight batching for the shard-DAG prover/verifier.
//!
//! Groups same-shape Projection weights into one MLE, commits each group once
//! (cacheable), then opens all of a model's weight claims in a single batch per
//! group. Split into three phases so the harness can time model commitment,
//! prove, and verify separately.
//!
//! Prover data is streamed to a per-group cache file and NOT kept resident:
//! `commit` only records the commitment and metadata, and the opening/verify
//! phases lazily restore one group's Merkle tree at a time. This keeps peak
//! memory proportional to a single group instead of the whole model, which is
//! what lets large MoE models run at all.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use sha2::Digest;
use zkie_core::common::field::Goldilocks;
use zkie_core::pcs::batch_open::{
    prove_batch_open_committed, verify_batch_open_committed, BatchOpeningProof,
};
use zkie_core::pcs::cache::{read_header, write_header, BatchCacheMetadata};
use zkie_core::pcs::committed::BatchCtx;
use zkie_core::pcs::whir::{Commitment, OpeningProtocol, Whir};
use zkie_ops::compose::{Op, ShardDagProof, Store, TensorData};

/// One committed group of same-shape projection weights. Holds only public
/// commitment data plus the parameters needed to rebuild the batch `Whir`;
/// witness and prover data are materialized lazily during opening.
pub struct WeightBatch {
    pub group_id: String,
    pub arity: usize,
    pub commitment: Commitment,
    pub protocol: OpeningProtocol,
    pub num_vars: usize,
    pub security: usize,
    pub pow_bits: usize,
    pub num_tables: usize,
    pub table_map: HashMap<usize, usize>,
    pub tensor_ids: Vec<usize>,
    pub cache_path: PathBuf,
}

/// A vertical batch opening over one weight group.
pub struct WeightOpening {
    pub group_idx: usize,
    pub table_idxs: Vec<usize>,
    pub points: Vec<Vec<Goldilocks>>,
    pub claimed: Vec<Goldilocks>,
    pub proof: BatchOpeningProof,
}

/// Phase 1: pack same-shape projection weights and commit each group once.
/// The Merkle prover data is streamed to a cache file and dropped; only the
/// commitment and rebuild parameters stay resident.
pub fn commit_weight_batches(
    store: &Store,
    ops: &[Op],
    group_size: usize,
    pow_bits: usize,
    model_id: &str,
) -> Vec<WeightBatch> {
    assert!(group_size > 0);
    let cache_dir = std::env::var("ZKIE_VERTICAL_CACHE_DIR")
        .unwrap_or_else(|_| format!("/data/jimmyshi/zkie-cache/{model_id}-model-batches-v2"));
    fs::create_dir_all(&cache_dir).expect("create vertical cache dir");

    let mut seen = HashSet::new();
    let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for op in ops {
        if let Op::Projection { w, .. } = op {
            if seen.insert(*w) {
                let len = match &store.v[*w] {
                    TensorData::Owned(values) => values.len(),
                    TensorData::Mmap { len, .. } => *len,
                };
                by_len.entry(len).or_default().push(*w);
            }
        }
    }

    let mut batches = Vec::new();
    for (elements, tensor_ids) in by_len {
        let arity = elements.trailing_zeros() as usize;
        assert_eq!(elements, 1usize << arity);
        // Horizontal merge: pack as many same-arity weights as the Whir PoW
        // budget allows into ONE BatchCtx, so a single FRI opening covers every
        // claim of that arity. num_vars = arity + log2(group_size); pow_bits=17
        // is enough for num_vars <= 26, so never pack beyond that.
        let max_group = if arity <= 26 {
            1usize << (26 - arity)
        } else {
            1
        };
        let eff_group_size = max_group.max(1);
        for (chunk_idx, ids) in tensor_ids.chunks(eff_group_size).enumerate() {
            let table_map: HashMap<usize, usize> =
                ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let table_data: Vec<_> = ids.iter().map(|&id| store.materialize(id)).collect();
            let ref_slices: Vec<&[Goldilocks]> =
                table_data.iter().map(|table| table.as_ref()).collect();
            let table_ids: Vec<String> =
                ids.iter().map(|id| format!("store-tensor-{id}")).collect();
            let source_bytes = bincode::serialize(&table_data).expect("serialize source tables");
            let source_digest: [u8; 32] = sha2::Sha256::digest(&source_bytes).into();
            let group_id = format!("projection-a{arity}-g{chunk_idx}");
            let metadata = BatchCacheMetadata::new(
                model_id.to_owned(),
                group_id.clone(),
                table_ids,
                arity as u32,
                ids.len() as u32,
                90,
                pow_bits as u32,
                source_digest,
            );
            let base_whir = Whir::with_params(arity, 90, pow_bits);
            let num_vars = (ids.len() * elements).next_power_of_two().trailing_zeros() as usize;
            let batch_whir = Whir::with_params(num_vars, 90, pow_bits);
            let protocol = batch_whir.batch_opening_protocol(ids.len(), arity);
            let cache_path =
                std::path::PathBuf::from(&cache_dir).join(format!("{group_id}.zkiecache"));

            let commitment = if cache_path.exists() {
                File::open(&cache_path)
                    .ok()
                    .and_then(|file| {
                        let mut reader = BufReader::new(file);
                        read_header(&mut reader)
                            .ok()
                            .and_then(|(found, commitment)| {
                                if found == metadata {
                                    Some(commitment)
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or_else(|| {
                        let (commitment, prover_data, _, _) = base_whir.commit_batch(&ref_slices);
                        write_cache(&cache_path, &metadata, &commitment, prover_data);
                        commitment
                    })
            } else {
                let (commitment, prover_data, _, _) = base_whir.commit_batch(&ref_slices);
                write_cache(&cache_path, &metadata, &commitment, prover_data);
                commitment
            };

            batches.push(WeightBatch {
                group_id,
                arity,
                commitment,
                protocol,
                num_vars,
                security: 90,
                pow_bits,
                num_tables: ids.len(),
                table_map,
                tensor_ids: ids.to_vec(),
                cache_path,
            });
        }
    }
    batches
}

fn write_cache(
    cache_path: &PathBuf,
    metadata: &BatchCacheMetadata,
    commitment: &Commitment,
    prover_data: zkie_core::pcs::whir::ProverData,
) {
    let temp_path = cache_path.with_extension("zkiecache.tmp");
    let file = File::create(&temp_path).expect("create packed cache");
    let mut writer = BufWriter::new(file);
    write_header(&mut writer, metadata, commitment).expect("write cache header");
    Whir::serialize_batch_prover_data_into(prover_data, &mut writer)
        .expect("serialize packed WHIR data");
    writer.flush().expect("flush packed cache");
    writer.get_ref().sync_all().expect("sync packed cache");
    drop(writer);
    fs::rename(&temp_path, cache_path).expect("publish packed cache");
}

/// Phase 2: open every weight claim emitted by the shard proof, batched per
/// group. Each group's Merkle tree is restored, used, and dropped one at a time.
pub fn prove_weight_openings(
    store: &Store,
    batches: &[WeightBatch],
    proof: &ShardDagProof,
) -> Vec<WeightOpening> {
    let mut openings = Vec::new();
    for (group_idx, batch) in batches.iter().enumerate() {
        let mut table_idxs = Vec::new();
        let mut points = Vec::new();
        let mut claimed = Vec::new();
        for shard in &proof.shards {
            for (tensor_id, point, value) in &shard.claims {
                if let Some(&table_idx) = batch.table_map.get(tensor_id) {
                    table_idxs.push(table_idx);
                    points.push(point.clone());
                    claimed.push(*value);
                }
            }
        }
        if table_idxs.is_empty() {
            continue;
        }

        let table_data: Vec<_> = batch
            .tensor_ids
            .iter()
            .map(|&id| store.materialize(id))
            .collect();
        let ref_slices: Vec<&[Goldilocks]> =
            table_data.iter().map(|table| table.as_ref()).collect();
        let request_tables: Vec<Vec<Goldilocks>> = table_idxs
            .iter()
            .map(|&table_idx| ref_slices[table_idx].to_vec())
            .collect();

        let whir = Whir::with_params(batch.num_vars, batch.security, batch.pow_bits);
        let mut reader = BufReader::new(File::open(&batch.cache_path).expect("open cache"));
        let (_, commitment) = read_header(&mut reader).expect("read cache header");
        let prover_data = whir
            .restore_batch_prover_data_from(&ref_slices, &mut reader)
            .expect("restore packed WHIR data");
        let ctx = BatchCtx {
            commitment,
            prover_data,
            protocol: batch.protocol.clone(),
            whir,
            num_tables: batch.num_tables,
        };
        let proof =
            prove_batch_open_committed(&ctx, &table_idxs, &request_tables, &points, &claimed)
                .expect("prove vertical weight opening");
        openings.push(WeightOpening {
            group_idx,
            table_idxs,
            points,
            claimed,
            proof,
        });
    }
    openings
}

/// Phase 3: verify every weight opening against its group commitment.
pub fn verify_weight_openings(batches: &[WeightBatch], openings: &[WeightOpening]) -> bool {
    for opening in openings {
        let batch = &batches[opening.group_idx];
        let whir = Whir::with_params(batch.num_vars, batch.security, batch.pow_bits);
        if !verify_batch_open_committed(
            &whir,
            &batch.commitment,
            &batch.protocol,
            batch.num_tables,
            &opening.table_idxs,
            &opening.points,
            &opening.claimed,
            &opening.proof,
        ) {
            return false;
        }
    }
    true
}
