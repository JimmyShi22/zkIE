
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
