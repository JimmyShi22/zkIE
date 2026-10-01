// GPU Poseidon2 Merkle commitment for zkIE (Goldilocks).
//
// Bit-compatible with p3-merkle-tree 0.7.0's MerkleTreeMmcs over
//   Poseidon2Goldilocks<16>,
//   MerkleHash   = PaddingFreeSponge<Perm, 16, 8, 8>,
//   MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>,
//   arity N = 2, DIGEST_ELEMS = 8.
// The permutation is the generic p3-poseidon2 Poseidon2 (sbox x^7, the
// 4x4-block-circulant "light" external MDS, and the width-16 Goldilocks
// internal diffusion), driven by the round constants zkie-gkr derives from
// SmallRng::seed_from_u64(1) and uploads through
// `zkie_p2_upload_constants` (the same derivation the CPU permutation uses,
// so device and host hashes agree bit for bit).
//
// Leaf geometry (p3-merkle-tree 0.7.0): one digest per matrix row.
// Rows are hashed in SIMD batches of 8: `vertically_packed_row` feeds the
// packed sponge lane l with row r0+l, absorbing 8 packed values (columns)
// per permutation. Tail rows (< 8 per batch, or < 8 rows total) are hashed
// scalar. Compression is binary (step 2), also SIMD-batched over 8 rows,
// with zero-digest padding handled by the Rust caller (padded layers are
// memset to zero and only rows < next_len are compressed).

#include <cstdint>
#include <ff/goldilocks.hpp>  // goldilocks::fr_t = gl64_t
#include <util/gpu_t.cuh>      // ngpus()

namespace {

using gl64_t = goldilocks::fr_t;

__constant__ uint64_t c_rc_ext_initial[4][16];
__constant__ uint64_t c_rc_ext_final[4][16];
__constant__ uint64_t c_rc_internal[22];
__constant__ uint64_t c_diag15;  // 1/2^32 mod p

// 2^{-1} mod (2^64 - 2^32 + 1) = (p + 1) / 2.
__device__ __forceinline__ gl64_t halve(const gl64_t& x)
{
    return x * gl64_t(0x7FFFFFFF80000001ULL);
}

__device__ __forceinline__ gl64_t gzero()
{
    gl64_t z;
    z.zero();
    return z;
}

// S-box x^7 via the x^2, x^3, x^4 addition chain used by p3.
__device__ __forceinline__ gl64_t sbox7(const gl64_t& x)
{
    gl64_t x2 = x;
    x2.sqr();
    gl64_t x3 = x2;
    x3 *= x;
    gl64_t x4 = x2;
    x4.sqr();
    return x3 * x4;
}

// 4x4 MDS block of the external "light" layer, [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]
// (p3-poseidon2 `apply_mat4`).
__device__ __forceinline__ void apply_mat4(gl64_t x[4])
{
    gl64_t t01 = x[0] + x[1];
    gl64_t t23 = x[2] + x[3];
    gl64_t t0123 = t01 + t23;
    gl64_t t01123 = t0123 + x[1];
    gl64_t t01233 = t0123 + x[3];
    gl64_t x0 = x[0];
    gl64_t x2 = x[2];
    x[3] = t01233 + x0 + x0;  // 3*x0 + x1 + x2 + 2*x3
    x[1] = t01123 + x2 + x2;  // x0 + 2*x1 + 3*x2 + x3
    x[0] = t01123 + t01;      // 2*x0 + 3*x1 + x2 + x3
    x[2] = t01233 + t23;      // x0 + x1 + 2*x2 + 3*x3
}

// External MDS for width 16 (p3-poseidon2 `mds_light_permutation`):
// apply the 4x4 block per 4-chunk, then add the per-congruence-class sums.
__device__ __forceinline__ void mds_light16(gl64_t st[16])
{
    apply_mat4(&st[0]);
    apply_mat4(&st[4]);
    apply_mat4(&st[8]);
    apply_mat4(&st[12]);

    gl64_t sums[4];
    sums[0] = gzero();
    sums[1] = gzero();
    sums[2] = gzero();
    sums[3] = gzero();
    for (int j = 0; j < 16; j += 4)
        for (int k = 0; k < 4; k++)
            sums[k] += st[j + k];
    for (int i = 0; i < 16; i++)
        st[i] += sums[i % 4];
}

// One external round: add round constants, S-box every element, MDS.
__device__ __forceinline__ void ext_round(gl64_t st[16], const uint64_t rc[16])
{
    for (int i = 0; i < 16; i++)
        st[i] += gl64_t(rc[i]);
    for (int i = 0; i < 16; i++)
        st[i] = sbox7(st[i]);
    mds_light16(st);
}

// One internal round: S-box state[0] only, then the width-16 Goldilocks
// diffusion (p3-goldilocks `internal_layer_mat_mul_goldilocks_16`).
__device__ __forceinline__ void int_round(gl64_t st[16], uint64_t rc)
{
    st[0] += gl64_t(rc);
    st[0] = sbox7(st[0]);

    gl64_t sum = gzero();
    for (int i = 0; i < 16; i++)
        sum += st[i];

    gl64_t two;
    two = st[0] + st[0];
    st[0] = sum - two;                       // V[0] = -2
    st[1] = sum + st[1];                     // V[1] = 1
    st[2] = sum + (st[2] + st[2]);           // V[2] = 2
    st[3] = sum + halve(st[3]);              // V[3] = 1/2
    two = st[4] + st[4];
    st[4] = sum + (two + st[4]);             // V[4] = 3
    two = st[5] + st[5];
    st[5] = sum + (two + two);               // V[5] = 4
    st[6] = sum - halve(st[6]);              // V[6] = -1/2
    two = st[7] + st[7];
    st[7] = sum - (two + st[7]);             // V[7] = -3
    two = st[8] + st[8];
    st[8] = sum - (two + two);               // V[8] = -4
    st[9] = sum + halve(halve(halve(st[9])));      // V[9] = 1/2^3
    st[10] = sum + halve(halve(halve(halve(st[10]))));      // V[10] = 1/2^4
    st[11] = sum + halve(halve(halve(halve(halve(st[11])))));  // V[11] = 1/2^5
    st[12] = sum - halve(halve(halve(st[12])));      // V[12] = -1/2^3
    st[13] = sum - halve(halve(halve(halve(st[13]))));      // V[13] = -1/2^4
    st[14] = sum - halve(halve(halve(halve(halve(st[14])))));  // V[14] = -1/2^5
    st[15] = sum + st[15] * gl64_t(c_diag15);   // V[15] = 1/2^32
}

// Full Poseidon2-16 permutation: initial light-MDS, 4 external rounds,
// 22 internal rounds, 4 external rounds.
__device__ __forceinline__ void p2_permute(gl64_t st[16])
{
    mds_light16(st);
    for (int r = 0; r < 4; r++)
        ext_round(st, c_rc_ext_initial[r]);
    for (int r = 0; r < 22; r++)
        int_round(st, c_rc_internal[r]);
    for (int r = 0; r < 4; r++)
        ext_round(st, c_rc_ext_final[r]);
}

// One sponge permutation on the 8 packed lanes of an 8-row batch.
// Lane l hashes row r0 + l; each block of 8 consecutive columns is one
// absorbed chunk, with one permutation per full block and one after a
// partial final block (PaddingFreeSponge overwrite semantics).
__device__ __forceinline__ void leaf_batch_lane(const uint64_t* __restrict__ mat,
                                                uint64_t* __restrict__ digests,
                                                uint32_t row, uint32_t w)
{
    gl64_t st[16];
    for (int i = 0; i < 16; i++)
        st[i] = gzero();

    uint32_t nblocks = w >> 3;
    for (uint32_t b = 0; b < nblocks; b++) {
        for (int i = 0; i < 8; i++)
            st[i] = gl64_t(mat[row * (uint64_t)w + (b << 3) + i]);
        p2_permute(st);
    }
    uint32_t rem = w & 7;
    if (rem) {
        for (uint32_t i = 0; i < rem; i++)
            st[i] = gl64_t(mat[row * (uint64_t)w + (nblocks << 3) + i]);
        p2_permute(st);
    }

    for (int e = 0; e < 8; e++)
        digests[row * (uint64_t)8 + e] = static_cast<uint64_t>(st[e]);
}

// Leaf digests for full 8-row batches (the packed hashing path).
__global__ void leaf_batch_kernel(const uint64_t* __restrict__ mat,
                                  uint64_t* __restrict__ digests,
                                  uint32_t h, uint32_t w)
{
    uint32_t r0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (r0 + 8 > h)
        return;
    for (uint32_t lane = 0; lane < 8; lane++)
        leaf_batch_lane(mat, digests, r0 + lane, w);
}

// Leaf digests for scalar rows (tail rows: the scalar hashing path).
__global__ void leaf_scalar_kernel(const uint64_t* __restrict__ mat,
                                   uint64_t* __restrict__ digests,
                                   uint32_t start_row, uint32_t h, uint32_t w)
{
    uint32_t row = start_row + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= h)
        return;
    leaf_batch_lane(mat, digests, row, w);  // same sponge, one lane
}

// One binary compression on the 8 packed lanes of an 8-row batch of the
// next layer: children are packed column-wise (p3 `compress`), the state is
// [child0(8), child1(8), 0(8)] and the first 8 outputs are the digests.
__device__ __forceinline__ void compress_batch_lane(const uint64_t* __restrict__ prev,
                                                    uint64_t* __restrict__ next,
                                                    uint32_t row, uint32_t step)
{
    gl64_t st[16];
    for (int n = 0; n < step; n++)
        for (int e = 0; e < 8; e++)
            st[n * 8 + e] = gl64_t(prev[(step * (uint64_t)row + n) * 8 + e]);
    for (int i = step * 8; i < 16; i++)
        st[i] = gzero();
    p2_permute(st);
    for (int e = 0; e < 8; e++)
        next[row * (uint64_t)8 + e] = static_cast<uint64_t>(st[e]);
}

__global__ void compress_batch_kernel(const uint64_t* __restrict__ prev,
                                      uint64_t* __restrict__ next,
                                      uint32_t next_len, uint32_t step)
{
    uint32_t r0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (r0 + 8 > next_len)
        return;
    for (uint32_t lane = 0; lane < 8; lane++)
        compress_batch_lane(prev, next, r0 + lane, step);
}

__global__ void compress_scalar_kernel(const uint64_t* __restrict__ prev,
                                       uint64_t* __restrict__ next,
                                       uint32_t from, uint32_t to, uint32_t step)
{
    uint32_t i = from + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= to)
        return;
    compress_batch_lane(prev, next, i, step);  // same compression, one lane
}

}  // namespace

extern "C" {

int zkie_p2_upload_constants(const uint64_t* rc_init, const uint64_t* rc_final,
                             const uint64_t* rc_internal, uint64_t diag15)
{
    cudaError_t e;
    e = cudaMemcpyToSymbol(c_rc_ext_initial, rc_init, 4 * 16 * sizeof(uint64_t));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_rc_ext_final, rc_final, 4 * 16 * sizeof(uint64_t));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_rc_internal, rc_internal, 22 * sizeof(uint64_t));
    if (e != cudaSuccess) return e;
    e = cudaMemcpyToSymbol(c_diag15, &diag15, sizeof(uint64_t));
    return e;
}

int zkie_p2_leaf_batch(const uint64_t* mat, uint64_t* digests, uint32_t h, uint32_t w)
{
    uint32_t batches = h / 8;
    if (batches == 0)
        return cudaSuccess;
    uint32_t threads = 256;
    uint32_t blocks = (batches + threads - 1) / threads;
    leaf_batch_kernel<<<blocks, threads>>>(mat, digests, h, w);
    return cudaGetLastError();
}

int zkie_p2_leaf_scalar(const uint64_t* mat, uint64_t* digests,
                        uint32_t start_row, uint32_t h, uint32_t w)
{
    if (start_row >= h)
        return cudaSuccess;
    uint32_t n = h - start_row;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    leaf_scalar_kernel<<<blocks, threads>>>(mat, digests, start_row, h, w);
    return cudaGetLastError();
}

int zkie_p2_compress_batch(const uint64_t* prev, uint64_t* next,
                           uint32_t next_len, uint32_t step)
{
    uint32_t batches = next_len / 8;
    if (batches == 0)
        return cudaSuccess;
    uint32_t threads = 256;
    uint32_t blocks = (batches + threads - 1) / threads;
    compress_batch_kernel<<<blocks, threads>>>(prev, next, next_len, step);
    return cudaGetLastError();
}

int zkie_p2_compress_scalar(const uint64_t* prev, uint64_t* next,
                            uint32_t from, uint32_t to, uint32_t step)
{
    if (from >= to)
        return cudaSuccess;
    uint32_t n = to - from;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    compress_scalar_kernel<<<blocks, threads>>>(prev, next, from, to, step);
    return cudaGetLastError();
}



int zkie_cuda_device_count()
{
    return static_cast<int>(ngpus());
}

}  // extern "C"
