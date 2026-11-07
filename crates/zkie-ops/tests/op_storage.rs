//! The same forward over resident values, scratch tensors and memory-mapped
//! files.

use std::fs;

use zkie_core::compute::{MemoryGovernor, MemoryRequest};
use zkie_core::storage::TensorStorageService;
use zkie_core::types::{
    Digest, ElementRange, MappedTensorId, TensorId, TensorSpec, TensorStorageRef, TensorView,
};
use zkie_engine::memory::BudgetedMemoryGovernor;
use zkie_engine::pcs::request::field::Fp3;
use zkie_engine::storage::mmap::MmapTensorReader;
use zkie_engine::storage::EngineStorage;

use zkie_ops::causal_attention::CausalAttentionOp;
use zkie_ops::matmul::MatMulOp;
use zkie_ops::moe_down::MoeDownOp;
use zkie_ops::op::Operation;
use zkie_ops::registry::{
    encode_i64, read_i64_view, spec_1d, wide_from_words, KernelRecord, NullProgress, PureForward,
    StorageOnlyPort, TempScratch, write_i64_scratch,
};
use zkie_ops::softmax::SoftmaxOp;

fn pure(op: &dyn PureForward, inputs: &[Vec<i64>]) -> KernelRecord {
    op.forward_values(inputs).unwrap()
}

fn run_views(
    op: &dyn Operation<Fp3>,
    storage: &EngineStorage,
    governor: &BudgetedMemoryGovernor,
    views: &[TensorView],
) -> (Vec<Vec<i64>>, Vec<Vec<i128>>) {
    let progress = NullProgress;
    let mut port = StorageOnlyPort::<Fp3>::new(storage, governor, &progress);
    let record = op.forward(&mut port, views).unwrap();
    let outputs = record
        .outputs
        .iter()
        .map(|view| read_i64_view(storage, view).unwrap())
        .collect();
    let raw = record
        .raw_accumulators
        .iter()
        .map(|view| {
            let words = read_i64_view(storage, view).unwrap();
            words
                .chunks_exact(2)
                .map(|pair| wide_from_words(pair[0], pair[1]))
                .collect()
        })
        .collect();
    (outputs, raw)
}

fn scratch_views(
    storage: &EngineStorage,
    governor: &BudgetedMemoryGovernor,
    inputs: &[Vec<i64>],
) -> Vec<TensorView> {
    inputs
        .iter()
        .map(|values| write_i64_scratch(storage, governor, &spec_1d(values.len()), values).unwrap())
        .collect()
}

fn mmap_views(
    temp: &TempScratch,
    storage: &mut EngineStorage,
    governor: &BudgetedMemoryGovernor,
    inputs: &[Vec<i64>],
) -> Vec<TensorView> {
    let mut views = Vec::with_capacity(inputs.len());
    for (index, values) in inputs.iter().enumerate() {
        let path = temp.path.join(format!("input-{index}.bin"));
        fs::write(&path, encode_i64(values)).unwrap();
        let spec = spec_1d(values.len());
        let permit = governor
            .acquire(&MemoryRequest {
                resident_bytes: 0,
                mapped_window_bytes: spec.byte_len().unwrap(),
                pinned_bytes: 0,
                metadata_bytes: 0,
                scratch_disk_bytes: 0,
            })
            .unwrap();
        let reader = MmapTensorReader::open(
            &path,
            spec.clone(),
            Digest([index as u8 + 1; 32]),
            permit,
        )
        .unwrap();
        let id = MappedTensorId(index as u64 + 1);
        storage.register_mapped(id, reader).unwrap();
        views.push(TensorView {
            tensor_id: TensorId(index as u64 + 100),
            spec,
            storage: TensorStorageRef::Mapped(id),
            element_range: ElementRange::new(0, values.len() as u64),
        });
    }
    views
}

fn harness<O: PureForward + Operation<Fp3>>(op: &O, inputs: &[Vec<i64>]) {
    let resident = pure(op, inputs);
    let temp = TempScratch::new("storage").unwrap();
    let governor = BudgetedMemoryGovernor::new(1 << 24);
    let storage = EngineStorage::new(&temp.path).unwrap();
    let scratch = scratch_views(&storage, &governor, inputs);
    let (scratch_outputs, scratch_raw) = run_views(op, &storage, &governor, &scratch);
    assert_eq!(scratch_outputs, resident.outputs);
    assert_eq!(scratch_raw, resident.raw_accumulators);

    let mut mapped_storage = EngineStorage::new(temp.path.join("mapped")).unwrap();
    let mapped = mmap_views(&temp, &mut mapped_storage, &governor, inputs);
    let (mapped_outputs, mapped_raw) =
        run_views(op, &mapped_storage, &governor, &mapped);
    assert_eq!(mapped_outputs, resident.outputs);
    assert_eq!(mapped_raw, resident.raw_accumulators);
    assert_eq!(mapped_outputs, scratch_outputs);

    let specs: Vec<TensorSpec> = inputs.iter().map(|values| spec_1d(values.len())).collect();
    let workspace = Operation::<Fp3>::workspace(op, zkie_ops::op::OpPhase::Forward, &specs).unwrap();
    assert!(workspace.scratch_disk_bytes > 0);
}

#[test]
fn matmul_forward_is_the_same_from_all_sources() {
    harness(
        &MatMulOp {
            m: 2,
            n: 2,
            k: 2,
            shift: 1,
        },
        &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
    );
}

#[test]
fn softmax_forward_is_the_same_from_all_sources() {
    harness(
        &SoftmaxOp { shift: 2 },
        &[vec![3, 1, 0, 2], vec![100, 50, 10, 5]],
    );
}

#[test]
fn moe_down_forward_is_the_same_from_all_sources() {
    harness(
        &MoeDownOp {
            shared_weights: vec![1, 0, 0, 1],
            routed_weights: vec![2, 0, 0, 2],
            width: 2,
            out_cols: 2,
            shift: 1,
        },
        &[vec![1, 2], vec![3, 4]],
    );
}

#[test]
fn causal_attention_forward_is_the_same_from_all_sources() {
    harness(
        &CausalAttentionOp {
            seq: 2,
            head_dim: 1,
            shift: 0,
        },
        &[vec![1, 2], vec![1, 1], vec![3, 4], vec![10, 5, 1]],
    );
}

#[test]
fn storage_outputs_are_sealed_scratch_and_permits_release() {
    let temp = TempScratch::new("sealed").unwrap();
    let governor = BudgetedMemoryGovernor::new(1 << 24);
    let storage = EngineStorage::new(&temp.path).unwrap();
    let views = scratch_views(&storage, &governor, &[vec![1, 2], vec![3, 4]]);
    assert_eq!(governor.snapshot().reserved_bytes, 0);
    let progress = NullProgress;
    let mut port = StorageOnlyPort::<Fp3>::new(&storage, &governor, &progress);
    let record = MatMulOp {
        m: 1,
        n: 1,
        k: 2,
        shift: 0,
    }
    .forward(&mut port, &views)
    .unwrap();
    assert_eq!(governor.snapshot().reserved_bytes, 0);
    let output = read_i64_view(&storage, &record.outputs[0]).unwrap();
    assert_eq!(output, vec![11]);
    assert!(matches!(
        record.outputs[0].storage,
        TensorStorageRef::Scratch(_)
    ));
}
