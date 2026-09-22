//! CUDA acceleration for the zkIE GKR/WHIR prover.
//!
//! Two backends, both bit-compatible with their CPU counterparts (which they
//! fall back to when no GPU is available or a kernel fails):
//!
//! - [`dft::CudaDft`] implements `p3_dft::TwoAdicSubgroupDft<Goldilocks>` on
//!   top of Sppark's Goldilocks NTT kernels (ct/gs mixed-radix, Plonky2/
//!   Plonky3-compatible roots). Only `dft_batch` is custom; every other
//!   method derives from it via the trait's defaults.
//! - [`merkle::CudaMerkleTreeMmcs`] implements `p3_commit::Mmcs<Goldilocks>`
//!   with Poseidon2-16 leaf hashing and compression on the GPU, producing the
//!   exact `p3_merkle_tree::MerkleTree` digest layers the CPU path produces
//!   (constructed through the vendored `MerkleTree::from_parts`), so
//!   openings and verification are the upstream CPU code.

#![allow(clippy::missing_safety_doc)]

pub mod dft;
pub mod merkle;

#[cfg(feature = "cuda")]
pub mod ffi {
    extern "C" {
        // goldilocks_ntt.cu
        pub fn zkie_cuda_device_count() -> i32;
        pub fn zkie_ntt_forward_goldilocks(
            d_mat: *mut u64,
            d_temp: *mut u64,
            lg_h: u32,
            w: u64,
            d_tw: *const u64,
        ) -> i32;
        // poseidon2_merkle.cu
        pub fn zkie_p2_upload_constants(
            rc_init: *const u64,
            rc_final: *const u64,
            rc_internal: *const u64,
            diag15: u64,
        ) -> i32;
        pub fn zkie_p2_leaf_batch(mat: *const u64, digests: *mut u64, h: u32, w: u32) -> i32;
        pub fn zkie_p2_leaf_scalar(
            mat: *const u64,
            digests: *mut u64,
            start_row: u32,
            h: u32,
            w: u32,
        ) -> i32;
        pub fn zkie_p2_compress_batch(
            prev: *const u64,
            next: *mut u64,
            next_len: u32,
            step: u32,
        ) -> i32;
        pub fn zkie_p2_compress_scalar(
            prev: *const u64,
            next: *mut u64,
            from: u32,
            to: u32,
            step: u32,
        ) -> i32;
        // CUDA runtime
        pub fn cudaMalloc(ptr: *mut *mut u64, size: usize) -> i32;
        pub fn cudaFree(ptr: *mut u64) -> i32;
        pub fn cudaMemcpy(dst: *mut u64, src: *const u64, size: usize, kind: i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaMemset(ptr: *mut u64, value: i32, size: usize) -> i32;
    }

    pub const CUDART_OK: i32 = 0;
    pub const MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const MEMCPY_DEVICE_TO_HOST: i32 = 2;

    /// Number of CUDA devices visible to this process (0 if CUDA is unusable).
    pub fn device_count() -> i32 {
        unsafe { zkie_cuda_device_count() }
    }
}

#[cfg(feature = "cuda")]
pub use ffi::device_count;

/// Whether a usable CUDA device is present.
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    device_count() > 0
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

#[cfg(feature = "cuda")]
pub mod buffer {
    use std::sync::Arc;

    use super::ffi::{cudaFree, cudaMalloc, CUDART_OK};

    /// A grow-only device allocation of `u64`s.
    pub struct CudaBuffer {
        ptr: *mut u64,
        len: usize,
    }

    // The raw pointer is a device allocation; the owner manages access via a
    // mutex and never hands it out beyond a single kernel call.
    unsafe impl Send for CudaBuffer {}
    unsafe impl Sync for CudaBuffer {}

    impl CudaBuffer {
        pub fn new(len: usize) -> Option<Self> {
            if len == 0 {
                return Some(Self {
                    ptr: std::ptr::null_mut(),
                    len: 0,
                });
            }
            let mut ptr: *mut u64 = std::ptr::null_mut();
            let rc = unsafe { cudaMalloc(&mut ptr, len * 8) };
            if rc != CUDART_OK || ptr.is_null() {
                return None;
            }
            Some(Self { ptr, len })
        }

        pub fn ptr(&self) -> *mut u64 {
            self.ptr
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn grow(&mut self, len: usize) -> Option<()> {
            if len <= self.len {
                return Some(());
            }
            let bigger = Self::new(len)?;
            *self = bigger;
            Some(())
        }
    }

    impl Drop for CudaBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { cudaFree(self.ptr) };
            }
        }
    }

    pub type SharedBuffer = Arc<std::sync::Mutex<CudaBuffer>>;
}
#[cfg(feature = "cuda")]
pub(crate) use buffer::SharedBuffer;
