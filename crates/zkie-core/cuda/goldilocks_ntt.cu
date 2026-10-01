// Goldilocks NTT for zkIE: purpose-built radix-2 DIT kernel.
//
// Implements p3-dft's contract (`TwoAdicSubgroupDft::dft_batch`) exactly:
// for each column, out[i] = P(omega_p3^i) in natural order, where omega_p3
// is Plonky3's two-adic generator. The twiddle tables are generated on the
// host from p3's own `TWO_ADIC_GENERATORS` and uploaded per size, so there
// is no root-of-unity convention to match — the device math is the textbook
// decimation-in-time butterfly network over gl64_t (canonical 64-bit
// arithmetic mod 2^64 - 2^32 + 1, byte-compatible with p3's Goldilocks).
//
// (The first implementation reused Sppark's mixed-radix NTT kernels, but
// their twiddle-table convention is incompatible with p3's generator — an
// impulse probe showed an effective generator of order 2 — so this kernel
// replaces them. Values stay partially reduced between stages; the final
// scatter canonicalizes.)
//
// Pipeline for an h x w row-major matrix (h = 2^lg):
//   1. gather:   temp[c*h + j] = mat[bitrev(j)*w + c]   (column-major,
//      bit-reversed input domain)
//   2. DIT stages (s = 1..lg): in-place butterflies on each column; the
//      network consumes bit-reversed input and produces natural-order
//      evaluations (out[k] = sum_j u[j] * omega^(k * bitrev(j)))
//   3. scatter:  mat[i*w + c] = canonical(temp[c*h + i])

#include <cstdint>
#include <ff/goldilocks.hpp>  // goldilocks::fr_t = gl64_t
#include <util/exception.cuh>  // CUDA_OK / cuda_error

namespace {

using gl64_t = goldilocks::fr_t;

// Gather columns into the bit-reversed input domain the DIT stages consume:
// temp[c*h + j] = mat[bitrev(j)*w + c].
__device__ __forceinline__ uint32_t bitrev32(uint32_t x, uint32_t bits)
{
    uint32_t r = 0;
    for (uint32_t b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

__global__ void gather_columns_kernel(const uint64_t* __restrict__ mat,
                                      uint64_t* __restrict__ temp,
                                      uint64_t h, uint64_t w, uint32_t lg)
{
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= h * w) return;
    uint64_t c = idx / h;
    uint64_t j = idx % h;
    uint64_t src_row = bitrev32((uint32_t)j, lg);
    temp[idx] = mat[src_row * w + c];
}

// One DIT stage (s in 1..=lg): butterflies between elements at distance
// 2^(s-1) within each column. `tw` holds 2^(s-1) twiddles:
// tw[j] = omega_p3^(j * 2^(lg-s)).
__global__ void ntt_dit_stage_kernel(uint64_t* __restrict__ d, uint32_t lg,
                                     uint64_t w, uint32_t s,
                                     const uint64_t* __restrict__ tw)
{
    const uint64_t n = (uint64_t)1 << lg;
    const uint64_t half = (uint64_t)1 << (s - 1);
    const uint64_t pairs = n >> 1;
    const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= pairs * w) return;

    const uint64_t col = tid / pairs;
    const uint64_t p = tid % pairs;
    const uint64_t i = (p / half) * (half << 1);
    const uint64_t j = p % half;
    const uint64_t base = col * n;

    gl64_t u = gl64_t(d[base + i + j]);
    gl64_t v = gl64_t(d[base + i + j + half]) * gl64_t(tw[j]);
    d[base + i + j] = (u + v)[0];
    d[base + i + j + half] = (u - v)[0];
}

// Scatter transformed columns back to row-major layout, canonicalizing
// (reducing mod p) the partially reduced values gl64_t keeps internally.
__global__ void scatter_canonical_columns_kernel(const uint64_t* __restrict__ temp,
                                                 uint64_t* __restrict__ mat,
                                                 uint64_t h, uint64_t w)
{
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= h * w) return;
    uint64_t c = idx / h;
    uint64_t i = idx % h;
    mat[i * w + c] = static_cast<uint64_t>(gl64_t(temp[idx]));
}

}  // namespace

extern "C" {

// Forward NTT over every column of a row-major h x w matrix held on the
// device (h = 2^lg_h, w arbitrary; lg_h == 0 or w == 0 is a no-op).
// `d_mat` is overwritten in place with the natural-order evaluations on
// Plonky3's two-adic generator; `d_temp` is a scratch buffer of h*w u64s.
// `d_tw` holds the concatenated per-stage twiddle tables: stage s (1..=lg)
// occupies 2^(s-1) u64s at offset 2^(s-1) - 2.
int zkie_ntt_forward_goldilocks(uint64_t* d_mat, uint64_t* d_temp,
                                uint32_t lg_h, uint64_t w,
                                const uint64_t* d_tw)
{
    if (lg_h == 0 || w == 0)
        return cudaSuccess;

    const uint64_t h = (uint64_t)1 << lg_h;
    const uint64_t total = h * w;
    const uint64_t threads = 256;

    try {
        const uint64_t blocks = (total + threads - 1) / threads;
        gather_columns_kernel<<<blocks, threads>>>(d_mat, d_temp, h, w, lg_h);
        CUDA_OK(cudaGetLastError());

        const uint64_t pairs = h >> 1;
        const uint64_t stage_blocks = (pairs * w + threads - 1) / threads;
        uint64_t tw_off = 0;
        for (uint32_t s = 1; s <= lg_h; s++) {
            ntt_dit_stage_kernel<<<stage_blocks, threads>>>(
                d_temp, lg_h, w, s, d_tw + tw_off);
            CUDA_OK(cudaGetLastError());
            tw_off += (uint64_t)1 << (s - 1);
        }

        scatter_canonical_columns_kernel<<<blocks, threads>>>(d_temp, d_mat, h, w);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        return e.code();
    }
    return cudaSuccess;
}

} // extern "C"
