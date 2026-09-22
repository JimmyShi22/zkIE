//! CUDA-backed `TwoAdicSubgroupDft<Goldilocks>`.
//!
//! `dft_batch` (the only method implemented; all other trait methods derive
//! from it) computes the DFT of each column of a row-major matrix: for each
//! column, natural-order evaluations at successive powers of p3's two-adic
//! generator. Internally: a purpose-built radix-2 DIT kernel over gl64_t
//! (canonical u64 arithmetic, byte-compatible with p3's Goldilocks), with
//! per-size twiddle tables generated on the host from p3's own
//! `TWO_ADIC_GENERATORS` and uploaded once per size.
//!
//! Falls back to `Radix2DFTSmallBatch` when CUDA is unavailable or a kernel
//! errors, so the CPU path remains fully usable on GPU-less machines.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

#[cfg(feature = "cuda")]
use crate::buffer::SharedBuffer;
#[cfg(feature = "cuda")]
use crate::ffi::{cudaDeviceSynchronize, cudaMemcpy, zkie_ntt_forward_goldilocks, CUDART_OK, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE};

#[cfg(feature = "cuda")]
const P: u64 = 0xffffffff00000001;

#[cfg(feature = "cuda")]
fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

/// Per-size DIT twiddle tables: for size 2^lg, stage s (1..=lg) contributes
/// 2^(s-1) entries, tw_s[j] = TWO_ADIC_GENERATORS[s]^j, concatenated in stage
/// order. Each size's table lives in its own 2^MAX_LG-slot of the device
/// twiddle buffer.
#[cfg(feature = "cuda")]
fn make_twiddles(lg: u32) -> Vec<u64> {
    const MAX_LG: u32 = 18;
    let slot: usize = 1 << MAX_LG;
    let mut out = vec![0u64; slot];
    let mut off = 0usize;
    for s in 1..=lg {
        let root = Goldilocks::TWO_ADIC_GENERATORS[s as usize].as_canonical_u64();
        let mut acc = 1u64; // root^0
        for j in 0..(1usize << (s - 1)) {
            out[off + j] = acc;
            acc = mul_mod(acc, root);
        }
        off += 1usize << (s - 1);
    }
    out
}

#[cfg(feature = "cuda")]
struct CudaNtt {
    /// Matrix + column-major scratch.
    buf: SharedBuffer,
    /// Fixed-size device buffer holding one twiddle slot per lg (2^18 u64s
    /// each); never reallocated, so offsets stay stable.
    tw_buf: SharedBuffer,
    uploaded: Mutex<HashMap<u32, ()>>,
}

#[cfg(feature = "cuda")]
impl CudaNtt {
    fn try_init() -> Option<Self> {
        const MAX_LG: u32 = 18;
        const SLOT: usize = 1 << MAX_LG;
        let buf = SharedBuffer::new(Mutex::new(crate::buffer::CudaBuffer::new(1 << 20)?));
        // MAX_LG+1 slots: lg 0..=18.
        let tw_buf = SharedBuffer::new(Mutex::new(crate::buffer::CudaBuffer::new(
            SLOT * (MAX_LG as usize + 1),
        )?));
        Some(Self {
            buf,
            tw_buf,
            uploaded: Mutex::new(HashMap::new()),
        })
    }

    /// Ensure the twiddle table for `lg` is on the device; returns the
    /// device pointer to its slot.
    fn ensure_twiddles(&self, lg: u32) -> Option<*mut u64> {
        const MAX_LG: u32 = 18;
        const SLOT: usize = 1 << MAX_LG;
        let mut uploaded = self.uploaded.lock().ok()?;
        if !uploaded.contains_key(&lg) {
            let table = make_twiddles(lg);
            let guard = self.tw_buf.lock().ok()?;
            let dst = unsafe { guard.ptr().add(lg as usize * SLOT) };
            let rc = unsafe { cudaMemcpy(dst, table.as_ptr(), SLOT * 8, MEMCPY_HOST_TO_DEVICE) };
            let rc = if rc == CUDART_OK {
                unsafe { cudaDeviceSynchronize() }
            } else {
                rc
            };
            if rc != CUDART_OK {
                return None;
            }
            uploaded.insert(lg, ());
        }
        let guard = self.tw_buf.lock().ok()?;
        Some(unsafe { guard.ptr().add(lg as usize * SLOT) })
    }
}

/// `TwoAdicSubgroupDft<Goldilocks>` on CUDA, falling back to the CPU
/// `Radix2DFTSmallBatch` when CUDA is unavailable.
#[derive(Clone, Default)]
pub struct CudaDft {
    #[cfg(feature = "cuda")]
    gpu: Option<Arc<CudaNtt>>,
}

impl CudaDft {
    /// Build the CUDA-backed DFT, probing for a usable device or falling
    /// back to CPU.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self {
                gpu: CudaNtt::try_init().map(Arc::new),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::default()
        }
    }

    /// Whether this instance is backed by a GPU.
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

impl TwoAdicSubgroupDft<Goldilocks> for CudaDft {
    type Evaluations = RowMajorMatrix<Goldilocks>;

    fn dft_batch(&self, mat: RowMajorMatrix<Goldilocks>) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();
        if h <= 1 {
            return mat;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (h, w);
            return Radix2DFTSmallBatch::new(h).dft_batch(mat);
        }
        #[cfg(feature = "cuda")]
        {
            match self.dft_batch_cuda(&mat) {
                Some(out) => out,
                None => Radix2DFTSmallBatch::new(h).dft_batch(mat),
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaDft {
    fn dft_batch_cuda(
        &self,
        mat: &RowMajorMatrix<Goldilocks>,
    ) -> Option<RowMajorMatrix<Goldilocks>> {
        let gpu = self.gpu.as_ref()?;
        let h = mat.height();
        let w = mat.width();
        let lg = h.trailing_zeros();
        assert_eq!(h, 1usize << lg, "height must be a power of two");
        let n = h * w;

        let d_tw = gpu.ensure_twiddles(lg)?;
        let vals: Vec<u64> = mat.values.iter().map(|f| f.as_canonical_u64()).collect();

        let mut guard = gpu.buf.lock().ok()?;
        guard.grow(n * 2)?; // matrix + column-major temp
        let d_mat = guard.ptr();
        let d_temp = unsafe { guard.ptr().add(n) };

        let rc = unsafe { cudaMemcpy(d_mat, vals.as_ptr(), n * 8, MEMCPY_HOST_TO_DEVICE) };
        if rc != CUDART_OK {
            return None;
        }
        let rc = unsafe { zkie_ntt_forward_goldilocks(d_mat, d_temp, lg, w as u64, d_tw) };
        if rc != CUDART_OK {
            return None;
        }

        let mut out = vec![0u64; n];
        let rc = unsafe { cudaMemcpy(out.as_mut_ptr(), d_mat, n * 8, MEMCPY_DEVICE_TO_HOST) };
        if rc != CUDART_OK {
            return None;
        }
        drop(guard);

        let values = out.into_iter().map(Goldilocks::new).collect();
        Some(RowMajorMatrix::new(values, w))
    }
}
