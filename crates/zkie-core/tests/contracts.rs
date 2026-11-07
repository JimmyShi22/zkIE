//! Interface contract tests for the core data and resource ports.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use zkie_core::common::codec::{decode_exact, Canonical, DecodeLimits};
use zkie_core::common::field::{Goldilocks, ProtocolField, GOLDILOCKS, GOLDILOCKS_EXT3, P};
use zkie_core::common::fixed_point::to_i64;
use zkie_core::compute::{MemoryPermit, MemoryRequest, PermitId};
use zkie_core::error::{Error, Result};
use zkie_core::storage::{TensorReader, TensorStorageService};
use zkie_core::types::{
    AllocationId, BufferGeneration, BufferHandle, BufferOwner, DType, Digest, ElementRange,
    LayoutSpec, MappedTensorId, ScratchTensorId, TensorId, TensorSpec, TensorStorageRef,
    TensorView, WorkspaceRequirement,
};

struct ByteReader {
    spec: TensorSpec,
}

impl TensorReader for ByteReader {
    fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    fn element_count(&self) -> u64 {
        self.spec.element_count().unwrap_or(0)
    }

    fn identity(&self) -> Digest {
        Digest::ZERO
    }

    fn read_range(
        &self,
        start_element: u64,
        element_count: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let element_size = self.spec.dtype.element_size() as usize;
        let expected = element_count
            .checked_mul(element_size)
            .ok_or(Error::InvalidIndex)?;
        if dst.len() != expected {
            return Err(Error::ShapeMismatch);
        }
        let end = start_element
            .checked_add(element_count as u64)
            .ok_or(Error::InvalidIndex)?;
        if end > self.element_count() {
            return Err(Error::InvalidIndex);
        }
        for (offset, byte) in dst.iter_mut().enumerate() {
            *byte = ((start_element as usize + offset) & 0xff) as u8;
        }
        Ok(())
    }
}

struct MockStorage;

impl TensorStorageService for MockStorage {
    fn reader(&self, view: &TensorView) -> Result<Arc<dyn TensorReader>> {
        view.validate()?;
        let reader: Arc<dyn TensorReader> = Arc::new(ByteReader {
            spec: view.spec.clone(),
        });
        Ok(reader)
    }

    fn create_scratch(
        &self,
        _spec: &TensorSpec,
        _permit: &MemoryPermit,
    ) -> Result<ScratchTensorId> {
        Err(Error::UnsupportedPrimitive)
    }

    fn write_scratch(
        &self,
        _id: ScratchTensorId,
        _range: ElementRange,
        _canonical_bytes: &[u8],
    ) -> Result<()> {
        Err(Error::UnsupportedPrimitive)
    }

    fn seal_scratch(&self, _id: ScratchTensorId) -> Result<TensorView> {
        Err(Error::UnsupportedPrimitive)
    }

    fn release_scratch(&self, _id: ScratchTensorId) -> Result<()> {
        Err(Error::UnsupportedPrimitive)
    }
}

fn read_view(service: &dyn TensorStorageService, view: &TensorView) -> Result<Vec<u8>> {
    let reader = service.reader(view)?;
    let bytes = view
        .element_range
        .len
        .checked_mul(view.spec.dtype.element_size())
        .ok_or(Error::InvalidIndex)?;
    let mut dst = vec![0u8; usize::try_from(bytes).map_err(|_| Error::InvalidIndex)?];
    reader.read_range(
        view.element_range.start,
        view.element_range.len as usize,
        &mut dst,
    )?;
    Ok(dst)
}

#[test]
fn same_operator_signature_for_resident_mapped_scratch() {
    let spec = TensorSpec::new(DType::I32, vec![4]);
    let views = [
        TensorView {
            tensor_id: TensorId(1),
            spec: spec.clone(),
            storage: TensorStorageRef::Resident(BufferHandle::new(
                BufferOwner(1),
                BufferGeneration(0),
                AllocationId(1),
                DType::I32,
                4,
            )),
            element_range: ElementRange::new(0, 4),
        },
        TensorView {
            tensor_id: TensorId(2),
            spec: spec.clone(),
            storage: TensorStorageRef::Mapped(MappedTensorId(2)),
            element_range: ElementRange::new(0, 4),
        },
        TensorView {
            tensor_id: TensorId(3),
            spec: spec.clone(),
            storage: TensorStorageRef::Scratch(ScratchTensorId(3)),
            element_range: ElementRange::new(0, 4),
        },
    ];
    let service = MockStorage;
    let resident = read_view(&service, &views[0]).unwrap();
    let mapped = read_view(&service, &views[1]).unwrap();
    let scratch = read_view(&service, &views[2]).unwrap();
    assert_eq!(resident, mapped);
    assert_eq!(resident, scratch);
    assert_eq!(resident.len(), 16);
}

#[test]
fn tensor_spec_arithmetic_is_checked() {
    let overflow = TensorSpec::new(DType::I32, vec![u64::MAX, 2]);
    assert!(overflow.element_count().is_err());

    let spec = TensorSpec::new(DType::I64, vec![2, 3]);
    assert_eq!(spec.element_count().unwrap(), 6);
    assert_eq!(spec.byte_len().unwrap(), 48);
    assert_eq!(spec.element_offset(&[1, 2]).unwrap(), 5);
    assert_eq!(
        ElementRange::new(1, 2).byte_range(&spec).unwrap(),
        (8, 16)
    );

    let mut column_major = spec.clone();
    column_major.layout = LayoutSpec::ColumnMajor;
    assert_eq!(column_major.element_offset(&[1, 2]).unwrap(), 5);

    let mut strided = spec.clone();
    strided.layout = LayoutSpec::Strided(vec![3, 1]);
    assert_eq!(strided.element_offset(&[1, 2]).unwrap(), 5);

    assert!(ElementRange::new(u64::MAX, 1).end().is_err());
    assert!(ElementRange::new(0, 7).validate(&spec).is_err());

    let mut padded = spec.clone();
    padded.padding.valid_len = Some(7);
    assert!(padded.validate().is_err());

    let workspace = WorkspaceRequirement {
        minimum_resident_bytes: 17,
        selected_resident_bytes: 16,
        ..WorkspaceRequirement::default()
    };
    assert!(matches!(workspace.validate(), Err(Error::ResourceLimit)));
}

#[test]
fn memory_permit_releases_only_on_drop() {
    let released = Arc::new(AtomicUsize::new(0));
    {
        let permit = MemoryPermit::new(
            PermitId(7),
            MemoryRequest {
                resident_bytes: 128,
                ..MemoryRequest::default()
            },
            {
                let released = Arc::clone(&released);
                move || {
                    released.fetch_add(1, Ordering::SeqCst);
                }
            },
        );
        assert_eq!(permit.id(), PermitId(7));
        assert_eq!(permit.request().resident_bytes, 128);
        assert_eq!(released.load(Ordering::SeqCst), 0);
    }
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn memory_request_total_is_checked() {
    let request = MemoryRequest {
        resident_bytes: u64::MAX,
        metadata_bytes: 1,
        ..MemoryRequest::default()
    };
    assert!(matches!(request.total_bytes(), Err(Error::ResourceLimit)));
}

#[test]
fn canonical_encoding_rejects_noncanonical_and_trailing() {
    let mut bytes = Vec::new();
    42u64.encode(&mut bytes).unwrap();
    let decoded: u64 = decode_exact(&bytes, &DecodeLimits::unlimited()).unwrap();
    assert_eq!(decoded, 42);

    let mut with_trailing = bytes.clone();
    with_trailing.push(0);
    assert!(decode_exact::<u64>(&with_trailing, &DecodeLimits::unlimited()).is_err());

    let noncanonical = P.to_le_bytes().to_vec();
    let mut input = noncanonical.as_slice();
    assert!(matches!(
        Goldilocks::decode(&mut input, &DecodeLimits::unlimited()),
        Err(Error::InvalidEncoding)
    ));

    let tight = DecodeLimits {
        max_bytes: 4,
        max_elements: 1,
    };
    assert!(decode_exact::<u64>(&bytes, &tight).is_err());
}

#[test]
fn bounded_integer_embedding_rejects_out_of_bound() {
    let value = Goldilocks::embed_bounded_integer(-7, 100).unwrap();
    assert_eq!(to_i64(value), -7);
    assert!(matches!(
        Goldilocks::embed_bounded_integer(101, 100),
        Err(Error::BoundViolation)
    ));
    assert!(matches!(
        Goldilocks::embed_bounded_integer(0, u128::MAX),
        Err(Error::BoundViolation)
    ));
}

#[test]
fn dtype_binds_field_descriptor() {
    assert_eq!(GOLDILOCKS.modulus, P);
    assert_eq!(GOLDILOCKS.extension_degree, 1);
    assert_eq!(GOLDILOCKS_EXT3.extension_degree, 3);
    assert_eq!(DType::BaseField(&GOLDILOCKS).element_size(), 8);
    assert_eq!(DType::ExtensionField(&GOLDILOCKS_EXT3).element_size(), 24);
    assert!(DType::BaseField(&GOLDILOCKS).field_descriptor().is_some());
    assert!(DType::I32.field_descriptor().is_none());
}
