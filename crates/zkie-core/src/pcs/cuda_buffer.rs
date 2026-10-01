
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
