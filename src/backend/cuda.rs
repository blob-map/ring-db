use crate::backend::{DiskSpec, RingComputeBackend};
use crate::error::{Result, RingDbError};
use crate::metadata::CandidateSetView;
use crate::query::Hit;
use std::cmp::Ordering;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const THREADS_PER_BLOCK: u32 = 256;
const CANDIDATES_ALL: u32 = 0;
const CANDIDATES_SPARSE: u32 = 1;
const CANDIDATES_DENSE: u32 = 2;
const HOST_SELECT_RATIO: usize = 64;

fn dense_candidate_count(words: &[u64], universe_len: usize) -> usize {
    let full_words = universe_len / 64;
    let mut count: usize = words[..full_words]
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum();
    let tail_bits = universe_len % 64;
    if tail_bits != 0 {
        let mask = (1_u64 << tail_bits) - 1;
        count += (words[full_words] & mask).count_ones() as usize;
    }
    count
}

fn compare_hits(left: &Hit, right: &Hit) -> Ordering {
    left.dist_sq
        .total_cmp(&right.dist_sq)
        .then_with(|| left.id.cmp(&right.id))
}

fn select_knn_on_host(k: usize, transfer_count: usize) -> bool {
    k.saturating_mul(HOST_SELECT_RATIO) >= transfer_count
}

#[cfg(windows)]
macro_rules! cuda_fn {
    ($(type $name:ident = fn($($arg:ty),* $(,)?) -> $ret:ty;)*) => {
        $(type $name = unsafe extern "system" fn($($arg),*) -> $ret;)*
    };
}

#[cfg(not(windows))]
macro_rules! cuda_fn {
    ($(type $name:ident = fn($($arg:ty),* $(,)?) -> $ret:ty;)*) => {
        $(type $name = unsafe extern "C" fn($($arg),*) -> $ret;)*
    };
}

cuda_fn! {
    type CuInit = fn(c_uint) -> CUresult;
    type CuDeviceGetCount = fn(*mut c_int) -> CUresult;
    type CuDeviceGet = fn(*mut CUdevice, c_int) -> CUresult;
    type CuCtxCreate = fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
    type CuCtxDestroy = fn(CUcontext) -> CUresult;
    type CuCtxSetCurrent = fn(CUcontext) -> CUresult;
    type CuCtxSynchronize = fn() -> CUresult;
    type CuMemAlloc = fn(*mut CUdeviceptr, usize) -> CUresult;
    type CuMemFree = fn(CUdeviceptr) -> CUresult;
    type CuMemcpyHtoD = fn(CUdeviceptr, *const c_void, usize) -> CUresult;
    type CuMemcpyDtoH = fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
    type CuModuleLoadData = fn(*mut CUmodule, *const c_void) -> CUresult;
    type CuModuleUnload = fn(CUmodule) -> CUresult;
    type CuModuleGetFunction = fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
    type CuLaunchKernel = fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult;
    type CuGetErrorName = fn(CUresult, *mut *const c_char) -> CUresult;
    type CuGetErrorString = fn(CUresult, *mut *const c_char) -> CUresult;
}

struct DynamicLibrary {
    handle: *mut c_void,
}

unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}

impl DynamicLibrary {
    fn open_cuda_driver() -> Result<Self> {
        #[cfg(windows)]
        const CANDIDATES: &[&[u8]] = &[b"nvcuda.dll\0"];

        #[cfg(target_os = "linux")]
        const CANDIDATES: &[&[u8]] = &[b"libcuda.so.1\0", b"libcuda.so\0"];

        #[cfg(all(unix, not(target_os = "linux")))]
        const CANDIDATES: &[&[u8]] = &[b"libcuda.dylib\0", b"libcuda.so\0"];

        let mut tried = Vec::new();
        for &name in CANDIDATES {
            let display = unsafe { CStr::from_ptr(name.as_ptr().cast()) }
                .to_string_lossy()
                .into_owned();
            match unsafe { Self::open(name.as_ptr().cast()) } {
                Ok(lib) => return Ok(lib),
                Err(err) => tried.push(format!("{display}: {err}")),
            }
        }

        Err(RingDbError::BackendUnavailable(format!(
            "could not load NVIDIA CUDA driver ({})",
            tried.join("; ")
        )))
    }

    unsafe fn symbol<T: Copy>(&self, name: &'static [u8]) -> Result<T> {
        let ptr = unsafe { platform_symbol(self.handle, name.as_ptr().cast()) };
        if ptr.is_null() {
            let name = unsafe { CStr::from_ptr(name.as_ptr().cast()) }.to_string_lossy();
            return Err(RingDbError::BackendUnavailable(format!(
                "CUDA driver symbol `{name}` is missing"
            )));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            platform_close(self.handle);
        }
    }
}

#[cfg(windows)]
impl DynamicLibrary {
    unsafe fn open(name: *const c_char) -> std::result::Result<Self, String> {
        let handle = unsafe { LoadLibraryA(name) };
        if handle.is_null() {
            return Err("LoadLibraryA returned null".to_string());
        }
        Ok(Self { handle })
    }
}

#[cfg(unix)]
impl DynamicLibrary {
    unsafe fn open(name: *const c_char) -> std::result::Result<Self, String> {
        let handle = unsafe { dlopen(name, RTLD_NOW) };
        if handle.is_null() {
            let err = unsafe { dlerror() };
            let msg = if err.is_null() {
                "dlopen returned null".to_string()
            } else {
                unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(msg);
        }
        Ok(Self { handle })
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(lp_lib_file_name: *const c_char) -> *mut c_void;
    fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
    fn FreeLibrary(h_lib_module: *mut c_void) -> i32;
}

#[cfg(windows)]
unsafe fn platform_symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    unsafe { GetProcAddress(handle, name) }
}

#[cfg(windows)]
unsafe fn platform_close(handle: *mut c_void) {
    unsafe {
        FreeLibrary(handle);
    }
}

#[cfg(unix)]
const RTLD_NOW: c_int = 2;

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[cfg(unix)]
unsafe fn platform_symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    unsafe { dlsym(handle, name) }
}

#[cfg(unix)]
unsafe fn platform_close(handle: *mut c_void) {
    unsafe {
        dlclose(handle);
    }
}

struct CudaDriver {
    _lib: DynamicLibrary,
    cu_init: CuInit,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_ctx_synchronize: CuCtxSynchronize,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    cu_get_error_name: CuGetErrorName,
    cu_get_error_string: CuGetErrorString,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    fn load() -> Result<Arc<Self>> {
        let lib = DynamicLibrary::open_cuda_driver()?;
        let driver = unsafe {
            Self {
                cu_init: lib.symbol(b"cuInit\0")?,
                cu_device_get_count: lib.symbol(b"cuDeviceGetCount\0")?,
                cu_device_get: lib.symbol(b"cuDeviceGet\0")?,
                cu_ctx_create: lib.symbol(b"cuCtxCreate_v2\0")?,
                cu_ctx_destroy: lib.symbol(b"cuCtxDestroy_v2\0")?,
                cu_ctx_set_current: lib.symbol(b"cuCtxSetCurrent\0")?,
                cu_ctx_synchronize: lib.symbol(b"cuCtxSynchronize\0")?,
                cu_mem_alloc: lib.symbol(b"cuMemAlloc_v2\0")?,
                cu_mem_free: lib.symbol(b"cuMemFree_v2\0")?,
                cu_memcpy_htod: lib.symbol(b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_dtoh: lib.symbol(b"cuMemcpyDtoH_v2\0")?,
                cu_module_load_data: lib.symbol(b"cuModuleLoadData\0")?,
                cu_module_unload: lib.symbol(b"cuModuleUnload\0")?,
                cu_module_get_function: lib.symbol(b"cuModuleGetFunction\0")?,
                cu_launch_kernel: lib.symbol(b"cuLaunchKernel\0")?,
                cu_get_error_name: lib.symbol(b"cuGetErrorName\0")?,
                cu_get_error_string: lib.symbol(b"cuGetErrorString\0")?,
                _lib: lib,
            }
        };

        unsafe {
            driver.check((driver.cu_init)(0), "cuInit")?;
            let mut count = 0;
            driver.check((driver.cu_device_get_count)(&mut count), "cuDeviceGetCount")?;
            if count == 0 {
                return Err(RingDbError::BackendUnavailable(
                    "CUDA driver loaded, but no CUDA devices were reported".to_string(),
                ));
            }
        }

        Ok(Arc::new(driver))
    }

    unsafe fn check(&self, result: CUresult, op: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }

        let mut name_ptr: *const c_char = ptr::null();
        let mut msg_ptr: *const c_char = ptr::null();
        let name = if unsafe { (self.cu_get_error_name)(result, &mut name_ptr) } == CUDA_SUCCESS
            && !name_ptr.is_null()
        {
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA_ERROR_{result}")
        };
        let msg = if unsafe { (self.cu_get_error_string)(result, &mut msg_ptr) } == CUDA_SUCCESS
            && !msg_ptr.is_null()
        {
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown CUDA driver error".to_string()
        };

        Err(RingDbError::Backend(format!("{op} failed: {name}: {msg}")))
    }
}

struct CudaContext {
    driver: Arc<CudaDriver>,
    raw: CUcontext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    fn create(driver: Arc<CudaDriver>) -> Result<Self> {
        unsafe {
            let mut device = 0;
            driver.check((driver.cu_device_get)(&mut device, 0), "cuDeviceGet")?;

            let mut raw = ptr::null_mut();
            driver.check(
                (driver.cu_ctx_create)(&mut raw, 0, device),
                "cuCtxCreate_v2",
            )?;
            Ok(Self { driver, raw })
        }
    }

    fn set_current(&self) -> Result<()> {
        unsafe {
            self.driver.check(
                (self.driver.cu_ctx_set_current)(self.raw),
                "cuCtxSetCurrent",
            )
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.raw);
            let _ = (self.driver.cu_ctx_destroy)(self.raw);
        }
    }
}

struct CudaModule {
    driver: Arc<CudaDriver>,
    raw: CUmodule,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    fn load(driver: Arc<CudaDriver>) -> Result<Self> {
        unsafe {
            let mut raw = ptr::null_mut();
            driver.check(
                (driver.cu_module_load_data)(&mut raw, CUDA_PTX.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            Ok(Self { driver, raw })
        }
    }

    fn function(&self, name: &'static [u8]) -> Result<CUfunction> {
        unsafe {
            let mut function = ptr::null_mut();
            self.driver.check(
                (self.driver.cu_module_get_function)(&mut function, self.raw, name.as_ptr().cast()),
                "cuModuleGetFunction",
            )?;
            Ok(function)
        }
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_module_unload)(self.raw);
        }
    }
}

struct DeviceBuffer<T> {
    driver: Arc<CudaDriver>,
    ptr: CUdeviceptr,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    fn from_slice(driver: Arc<CudaDriver>, data: &[T]) -> Result<Self> {
        let buffer = Self::uninit(driver, data.len())?;
        buffer.copy_from_slice(data)?;
        Ok(buffer)
    }

    fn copy_from_slice(&self, data: &[T]) -> Result<()> {
        assert!(data.len() <= self.len);
        if !data.is_empty() {
            unsafe {
                self.driver.check(
                    (self.driver.cu_memcpy_htod)(
                        self.ptr,
                        data.as_ptr().cast(),
                        std::mem::size_of_val(data),
                    ),
                    "cuMemcpyHtoD_v2",
                )?;
            }
        }
        Ok(())
    }

    fn copy_from_slice_at(&self, offset: usize, data: &[T]) -> Result<()> {
        assert!(offset <= self.len);
        assert!(data.len() <= self.len - offset);
        if !data.is_empty() {
            let byte_offset = offset
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| RingDbError::Backend("CUDA copy offset overflow".to_string()))?;
            let dst = self.ptr.checked_add(byte_offset as u64).ok_or_else(|| {
                RingDbError::Backend("CUDA destination pointer overflow".to_string())
            })?;
            unsafe {
                self.driver.check(
                    (self.driver.cu_memcpy_htod)(
                        dst,
                        data.as_ptr().cast(),
                        std::mem::size_of_val(data),
                    ),
                    "cuMemcpyHtoD_v2",
                )?;
            }
        }
        Ok(())
    }

    fn copy_to_vec(&self, len: usize) -> Result<Vec<T>> {
        self.copy_to_vec_at(0, len)
    }

    fn copy_to_vec_at(&self, offset: usize, len: usize) -> Result<Vec<T>> {
        assert!(offset <= self.len);
        assert!(len <= self.len - offset);
        let mut out: Vec<T> = Vec::with_capacity(len);
        if len > 0 {
            let byte_offset = offset
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| RingDbError::Backend("CUDA copy offset overflow".to_string()))?;
            let src = self
                .ptr
                .checked_add(byte_offset as u64)
                .ok_or_else(|| RingDbError::Backend("CUDA source pointer overflow".to_string()))?;
            unsafe {
                self.driver.check(
                    (self.driver.cu_memcpy_dtoh)(
                        out.as_mut_ptr().cast(),
                        src,
                        len * std::mem::size_of::<T>(),
                    ),
                    "cuMemcpyDtoH_v2",
                )?;
                out.set_len(len);
            }
        }
        Ok(out)
    }
}

enum CandidateBuffer {
    All,
    Sparse(DeviceBuffer<u32>),
    Dense(DeviceBuffer<u64>),
}

struct PreparedCandidates {
    buffer: CandidateBuffer,
    work_count: u32,
    eligible_count: usize,
}

impl PreparedCandidates {
    fn mode(&self) -> u32 {
        match &self.buffer {
            CandidateBuffer::All => CANDIDATES_ALL,
            CandidateBuffer::Sparse(_) => CANDIDATES_SPARSE,
            CandidateBuffer::Dense(_) => CANDIDATES_DENSE,
        }
    }

    fn ptr(&self) -> CUdeviceptr {
        match &self.buffer {
            CandidateBuffer::All => 0,
            CandidateBuffer::Sparse(ids) => ids.ptr,
            CandidateBuffer::Dense(words) => words.ptr,
        }
    }
}

impl<T> DeviceBuffer<T> {
    fn uninit(driver: Arc<CudaDriver>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| RingDbError::Backend("CUDA allocation size overflow".to_string()))?;
        let mut ptr = 0;
        if bytes > 0 {
            unsafe {
                driver.check((driver.cu_mem_alloc)(&mut ptr, bytes), "cuMemAlloc_v2")?;
            }
        }
        Ok(Self {
            driver,
            ptr,
            len,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
}

/// CUDA brute-force backend.
///
/// The dataset is copied to VRAM once during `upload_f32_dataset`. Each query
/// launches one CUDA thread per vector, computes its distance on-device, and
/// compacts hits into output buffers via a device-side atomic counter.
pub struct CudaBackend {
    vectors: Option<DeviceBuffer<f32>>,
    out_hits: Option<DeviceBuffer<Hit>>,
    out_count: Option<DeviceBuffer<u32>>,
    _module: CudaModule,
    context: CudaContext,
    driver: Arc<CudaDriver>,
    launch_lock: Mutex<()>,
    dims: usize,
    n_vectors: usize,
    ring_kernel: CUfunction,
    disk_kernel: CUfunction,
    weighted_ring_kernel: CUfunction,
    weighted_disk_kernel: CUfunction,
    disk_intersection_kernel: CUfunction,
    disk_intersection3_kernel: CUfunction,
    mixed_disk_intersection_kernel: CUfunction,
    knn_distance_kernel: CUfunction,
    knn_argmin_kernel: CUfunction,
}

unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        let driver = CudaDriver::load()?;
        let context = CudaContext::create(driver.clone())?;
        let module = CudaModule::load(driver.clone())?;
        let ring_kernel = module.function(b"ring_query_f32_kernel\0")?;
        let disk_kernel = module.function(b"disk_query_f32_kernel\0")?;
        let weighted_ring_kernel = module.function(b"weighted_ring_query_f32_kernel\0")?;
        let weighted_disk_kernel = module.function(b"weighted_disk_query_f32_kernel\0")?;
        let disk_intersection_kernel = module.function(b"disk_intersection_query_f32_kernel\0")?;
        let disk_intersection3_kernel =
            module.function(b"disk_intersection3_query_f32_kernel\0")?;
        let mixed_disk_intersection_kernel =
            module.function(b"mixed_disk_intersection_query_f32_kernel\0")?;
        let knn_distance_kernel = module.function(b"knn_distance_f32_kernel\0")?;
        let knn_argmin_kernel = module.function(b"knn_argmin_f32_kernel\0")?;

        Ok(Self {
            vectors: None,
            out_hits: None,
            out_count: None,
            _module: module,
            context,
            driver,
            launch_lock: Mutex::new(()),
            dims: 0,
            n_vectors: 0,
            ring_kernel,
            disk_kernel,
            weighted_ring_kernel,
            weighted_disk_kernel,
            disk_intersection_kernel,
            disk_intersection3_kernel,
            mixed_disk_intersection_kernel,
            knn_distance_kernel,
            knn_argmin_kernel,
        })
    }

    fn require_uploaded(&self) -> Result<&DeviceBuffer<f32>> {
        self.vectors
            .as_ref()
            .ok_or_else(|| RingDbError::Backend("CUDA dataset has not been uploaded".to_string()))
    }

    fn require_output_buffers(&self) -> Result<(&DeviceBuffer<Hit>, &DeviceBuffer<u32>)> {
        let out_hits = self.out_hits.as_ref().ok_or_else(|| {
            RingDbError::Backend("CUDA output hits are not allocated".to_string())
        })?;
        let out_count = self.out_count.as_ref().ok_or_else(|| {
            RingDbError::Backend("CUDA output counter is not allocated".to_string())
        })?;
        Ok((out_hits, out_count))
    }

    fn validate_query(&self, dims: usize, query: &[f32]) -> Result<()> {
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }
        if dims != self.dims {
            return Err(RingDbError::DimensionMismatch {
                expected: self.dims,
                got: dims,
            });
        }
        Ok(())
    }

    fn validate_weights(dims: usize, weights: &[f32]) -> Result<()> {
        if weights.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: weights.len(),
            });
        }
        Ok(())
    }

    fn upload_transposed_vectors(
        &self,
        vectors: &[f32],
        dims: usize,
        n_vectors: usize,
    ) -> Result<DeviceBuffer<f32>> {
        let buffer = DeviceBuffer::uninit(self.driver.clone(), vectors.len())?;
        let mut column = vec![0.0f32; n_vectors];
        for dim in 0..dims {
            for (dst, row) in column.iter_mut().zip(vectors.chunks_exact(dims)) {
                *dst = row[dim];
            }
            buffer.copy_from_slice_at(dim * n_vectors, &column)?;
        }
        Ok(buffer)
    }

    fn prepare_candidates(
        &self,
        candidates: CandidateSetView<'_>,
    ) -> Result<Option<PreparedCandidates>> {
        let prepared = match candidates {
            CandidateSetView::Empty => return Ok(None),
            CandidateSetView::All => PreparedCandidates {
                buffer: CandidateBuffer::All,
                work_count: self.n_vectors as u32,
                eligible_count: self.n_vectors,
            },
            CandidateSetView::Sparse(ids) => {
                if ids.is_empty() {
                    return Ok(None);
                }
                if ids.last().is_some_and(|&id| id as usize >= self.n_vectors) {
                    return Err(RingDbError::Backend(
                        "CUDA sparse candidate ID is outside the dataset".to_string(),
                    ));
                }
                PreparedCandidates {
                    buffer: CandidateBuffer::Sparse(DeviceBuffer::from_slice(
                        self.driver.clone(),
                        ids,
                    )?),
                    work_count: ids.len() as u32,
                    eligible_count: ids.len(),
                }
            }
            CandidateSetView::Dense(words) => {
                let expected_words = self.n_vectors.div_ceil(64);
                if words.len() != expected_words {
                    return Err(RingDbError::Backend(format!(
                        "CUDA dense candidate bitmap has {} words, expected {expected_words}",
                        words.len()
                    )));
                }
                let eligible_count = dense_candidate_count(words, self.n_vectors);
                if eligible_count == 0 {
                    return Ok(None);
                }
                PreparedCandidates {
                    buffer: CandidateBuffer::Dense(DeviceBuffer::from_slice(
                        self.driver.clone(),
                        words,
                    )?),
                    work_count: self.n_vectors as u32,
                    eligible_count,
                }
            }
        };
        Ok(Some(prepared))
    }

    fn launch_ring(
        &self,
        query: &[f32],
        lower_sq: f32,
        upper_sq: f32,
        disk_mode: bool,
        candidate_view: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 || matches!(candidate_view, CandidateSetView::Empty) {
            return Ok(Vec::new());
        }
        if self.n_vectors > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX vectors".to_string(),
            ));
        }
        if self.dims > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX dimensions".to_string(),
            ));
        }

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let vectors = self.require_uploaded()?;
        let (out_hits, out_count) = self.require_output_buffers()?;
        let Some(candidates) = self.prepare_candidates(candidate_view)? else {
            return Ok(Vec::new());
        };
        let query_buf = DeviceBuffer::from_slice(self.driver.clone(), query)?;
        let zero = [0u32];
        out_count.copy_from_slice(&zero)?;

        let mut vectors_ptr = vectors.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut candidate_mode = candidates.mode();
        let mut candidate_ptr = candidates.ptr();
        let mut work_count = candidates.work_count;
        let mut query_ptr = query_buf.ptr;
        let mut lower_sq = lower_sq;
        let mut upper_sq = upper_sq;
        let mut out_hits_ptr = out_hits.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = work_count.div_ceil(THREADS_PER_BLOCK);
        let mut ring_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut lower_sq as *mut f32).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];
        let mut disk_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            let (kernel, params) = if disk_mode {
                (self.disk_kernel, disk_params.as_mut_ptr())
            } else {
                (self.ring_kernel, ring_params.as_mut_ptr())
            };
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    kernel,
                    grid_x,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params,
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
        }

        let hit_count = out_count.copy_to_vec(1)?[0] as usize;
        let hit_count = hit_count.min(candidates.eligible_count);
        out_hits.copy_to_vec(hit_count)
    }

    fn launch_weighted_ring(
        &self,
        query: &[f32],
        weights: &[f32],
        lower_sq: f32,
        upper_sq: f32,
        disk_mode: bool,
        candidate_view: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 || matches!(candidate_view, CandidateSetView::Empty) {
            return Ok(Vec::new());
        }
        if self.n_vectors > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX vectors".to_string(),
            ));
        }
        if self.dims > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX dimensions".to_string(),
            ));
        }

        let packed_len = query
            .len()
            .checked_add(weights.len())
            .ok_or_else(|| RingDbError::Backend("CUDA query packing overflow".to_string()))?;
        let mut query_and_weights = Vec::with_capacity(packed_len);
        query_and_weights.extend_from_slice(query);
        query_and_weights.extend_from_slice(weights);

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let vectors = self.require_uploaded()?;
        let (out_hits, out_count) = self.require_output_buffers()?;
        let Some(candidates) = self.prepare_candidates(candidate_view)? else {
            return Ok(Vec::new());
        };
        let query_buf = DeviceBuffer::from_slice(self.driver.clone(), &query_and_weights)?;
        out_count.copy_from_slice(&[0u32])?;

        let weights_byte_offset = query
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| RingDbError::Backend("CUDA query offset overflow".to_string()))?;
        let mut vectors_ptr = vectors.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut candidate_mode = candidates.mode();
        let mut candidate_ptr = candidates.ptr();
        let mut work_count = candidates.work_count;
        let mut query_ptr = query_buf.ptr;
        let mut weights_ptr = query_buf
            .ptr
            .checked_add(weights_byte_offset as u64)
            .ok_or_else(|| RingDbError::Backend("CUDA query pointer overflow".to_string()))?;
        let mut lower_sq = lower_sq;
        let mut upper_sq = upper_sq;
        let mut out_hits_ptr = out_hits.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = work_count.div_ceil(THREADS_PER_BLOCK);
        let mut ring_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut weights_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut lower_sq as *mut f32).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];
        let mut disk_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut weights_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            let (kernel, params) = if disk_mode {
                (self.weighted_disk_kernel, disk_params.as_mut_ptr())
            } else {
                (self.weighted_ring_kernel, ring_params.as_mut_ptr())
            };
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    kernel,
                    grid_x,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params,
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
        }

        let hit_count = out_count.copy_to_vec(1)?[0] as usize;
        let hit_count = hit_count.min(candidates.eligible_count);
        out_hits.copy_to_vec(hit_count)
    }

    fn validate_disks(&self, dims: usize, disks: &[DiskSpec<'_>]) -> Result<()> {
        if dims != self.dims {
            return Err(RingDbError::DimensionMismatch {
                expected: self.dims,
                got: dims,
            });
        }
        if disks.len() > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX disks".to_string(),
            ));
        }
        for disk in disks {
            if disk.query.len() != dims {
                return Err(RingDbError::DimensionMismatch {
                    expected: dims,
                    got: disk.query.len(),
                });
            }
            if let Some(weights) = disk.weights {
                Self::validate_weights(dims, weights)?;
            }
        }
        Ok(())
    }

    fn launch_disk_intersection(
        &self,
        disks: &[DiskSpec<'_>],
        candidate_view: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0
            || disks.is_empty()
            || matches!(candidate_view, CandidateSetView::Empty)
        {
            return Ok(Vec::new());
        }
        if self.n_vectors > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX vectors".to_string(),
            ));
        }
        if self.dims > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX dimensions".to_string(),
            ));
        }

        let mut disk_queries = Vec::with_capacity(disks.len() * self.dims);
        let mut disk_upper_sq = Vec::with_capacity(disks.len());
        for disk in disks {
            debug_assert!(disk.weights.is_none());
            disk_queries.extend_from_slice(disk.query);
            disk_upper_sq.push(disk.d_max * disk.d_max);
        }

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let vectors = self.require_uploaded()?;
        let (out_hits, out_count) = self.require_output_buffers()?;
        let Some(candidates) = self.prepare_candidates(candidate_view)? else {
            return Ok(Vec::new());
        };
        let disk_queries_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_queries)?;
        let disk_upper_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_upper_sq)?;
        out_count.copy_from_slice(&[0u32])?;

        let mut vectors_ptr = vectors.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut candidate_mode = candidates.mode();
        let mut candidate_ptr = candidates.ptr();
        let mut work_count = candidates.work_count;
        let mut disk_queries_ptr = disk_queries_buf.ptr;
        let mut disk_upper_ptr = disk_upper_buf.ptr;
        let mut n_disks_u32 = disks.len() as u32;
        let mut out_hits_ptr = out_hits.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = work_count.div_ceil(THREADS_PER_BLOCK);
        // The warp-compacting three-disk kernel requires lane zero to remain
        // active. Dense masks can create holes within a warp, so they use the
        // generic kernel; All and sparse work lists stay contiguous.
        let kernel = if disks.len() == 3 && !matches!(candidate_view, CandidateSetView::Dense(_)) {
            self.disk_intersection3_kernel
        } else {
            self.disk_intersection_kernel
        };
        let mut params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut disk_queries_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut disk_upper_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut n_disks_u32 as *mut u32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    kernel,
                    grid_x,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
        }

        let hit_count = out_count.copy_to_vec(1)?[0] as usize;
        let hit_count = hit_count.min(candidates.eligible_count);
        out_hits.copy_to_vec(hit_count)
    }

    fn launch_mixed_disk_intersection(
        &self,
        disks: &[DiskSpec<'_>],
        candidate_view: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0
            || disks.is_empty()
            || matches!(candidate_view, CandidateSetView::Empty)
        {
            return Ok(Vec::new());
        }
        if self.n_vectors > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX vectors".to_string(),
            ));
        }
        if self.dims > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX dimensions".to_string(),
            ));
        }

        let query_values_len = disks
            .len()
            .checked_mul(self.dims)
            .ok_or_else(|| RingDbError::Backend("CUDA query packing overflow".to_string()))?;
        let weighted_disks = disks.iter().filter(|disk| disk.weights.is_some()).count();
        let weight_values_len = weighted_disks
            .checked_mul(self.dims)
            .ok_or_else(|| RingDbError::Backend("CUDA weight packing overflow".to_string()))?;
        if weight_values_len > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA packed weights exceed the u32 offset range".to_string(),
            ));
        }
        let packed_values_len = query_values_len
            .checked_add(weight_values_len)
            .ok_or_else(|| RingDbError::Backend("CUDA query packing overflow".to_string()))?;

        let mut disk_data = Vec::with_capacity(packed_values_len);
        for disk in disks {
            disk_data.extend_from_slice(disk.query);
        }
        let weights_start = disk_data.len();

        // Offsets count f32 elements from the packed weight suffix. The
        // sentinel keeps the metric decision outside both dimension loops.
        let mut weight_offsets = Vec::with_capacity(disks.len());
        for disk in disks {
            if let Some(weights) = disk.weights {
                let offset = disk_data.len() - weights_start;
                let offset = u32::try_from(offset).map_err(|_| {
                    RingDbError::Backend(
                        "CUDA packed weights exceed the u32 offset range".to_string(),
                    )
                })?;
                weight_offsets.push(offset);
                disk_data.extend_from_slice(weights);
            } else {
                weight_offsets.push(u32::MAX);
            }
        }

        let disk_upper_sq: Vec<f32> = disks.iter().map(|disk| disk.d_max * disk.d_max).collect();

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let vectors = self.require_uploaded()?;
        let (out_hits, out_count) = self.require_output_buffers()?;
        let Some(candidates) = self.prepare_candidates(candidate_view)? else {
            return Ok(Vec::new());
        };
        let disk_data_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_data)?;
        let disk_upper_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_upper_sq)?;
        let weight_offsets_buf = DeviceBuffer::from_slice(self.driver.clone(), &weight_offsets)?;
        out_count.copy_from_slice(&[0u32])?;

        let weights_byte_offset = weights_start
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| RingDbError::Backend("CUDA weight pointer overflow".to_string()))?;
        let mut vectors_ptr = vectors.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut candidate_mode = candidates.mode();
        let mut candidate_ptr = candidates.ptr();
        let mut work_count = candidates.work_count;
        let mut disk_queries_ptr = disk_data_buf.ptr;
        let mut disk_weights_ptr = disk_data_buf
            .ptr
            .checked_add(weights_byte_offset as u64)
            .ok_or_else(|| RingDbError::Backend("CUDA weight pointer overflow".to_string()))?;
        let mut disk_upper_ptr = disk_upper_buf.ptr;
        let mut weight_offsets_ptr = weight_offsets_buf.ptr;
        let mut n_disks_u32 = disks.len() as u32;
        let mut out_hits_ptr = out_hits.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = work_count.div_ceil(THREADS_PER_BLOCK);
        let mut params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut disk_queries_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut disk_weights_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut disk_upper_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut weight_offsets_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut n_disks_u32 as *mut u32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    self.mixed_disk_intersection_kernel,
                    grid_x,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
        }

        let hit_count = out_count.copy_to_vec(1)?[0] as usize;
        let hit_count = hit_count.min(candidates.eligible_count);
        out_hits.copy_to_vec(hit_count)
    }

    fn launch_knn(
        &self,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        candidate_view: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 || k == 0 || matches!(candidate_view, CandidateSetView::Empty) {
            return Ok(Vec::new());
        }
        if self.dims > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX dimensions".to_string(),
            ));
        }

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let vectors = self.require_uploaded()?;
        let (out_hits, _) = self.require_output_buffers()?;
        let Some(candidates) = self.prepare_candidates(candidate_view)? else {
            return Ok(Vec::new());
        };
        let target_count = k.min(candidates.eligible_count);
        let query_buf = DeviceBuffer::from_slice(self.driver.clone(), query)?;
        let weights_buf = weights
            .map(|values| DeviceBuffer::from_slice(self.driver.clone(), values))
            .transpose()?;

        let mut vectors_ptr = vectors.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut candidate_mode = candidates.mode();
        let mut candidate_ptr = candidates.ptr();
        let mut work_count = candidates.work_count;
        let mut query_ptr = query_buf.ptr;
        let mut weights_ptr = weights_buf.as_ref().map_or(0, |buffer| buffer.ptr);
        let mut weighted_mode = u32::from(weights_buf.is_some());
        let mut out_hits_ptr = out_hits.ptr;
        let grid_x = work_count.div_ceil(THREADS_PER_BLOCK);
        let mut params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut candidate_mode as *mut u32).cast::<c_void>(),
            (&mut candidate_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut work_count as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut weights_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut weighted_mode as *mut u32).cast::<c_void>(),
            (&mut out_hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    self.knn_distance_kernel,
                    grid_x,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(knn distance)",
            )?;
            self.driver
                .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
        }

        if select_knn_on_host(target_count, candidates.work_count as usize) {
            let mut hits = out_hits.copy_to_vec(candidates.work_count as usize)?;
            hits.retain(|hit| hit.id != u32::MAX);
            hits.sort_unstable_by(compare_hits);
            hits.truncate(target_count);
            return Ok(hits);
        }

        let best = DeviceBuffer::from_slice(self.driver.clone(), &[u64::MAX])?;
        let mut hits = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            best.copy_from_slice(&[u64::MAX])?;
            let mut hits_ptr = out_hits.ptr;
            let mut best_ptr = best.ptr;
            let mut argmin_params = [
                (&mut hits_ptr as *mut CUdeviceptr).cast::<c_void>(),
                (&mut work_count as *mut u32).cast::<c_void>(),
                (&mut best_ptr as *mut CUdeviceptr).cast::<c_void>(),
            ];
            unsafe {
                self.driver.check(
                    (self.driver.cu_launch_kernel)(
                        self.knn_argmin_kernel,
                        grid_x,
                        1,
                        1,
                        THREADS_PER_BLOCK,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        argmin_params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(knn argmin)",
                )?;
                self.driver
                    .check((self.driver.cu_ctx_synchronize)(), "cuCtxSynchronize")?;
            }

            let packed = best.copy_to_vec(1)?[0];
            if packed == u64::MAX {
                return Err(RingDbError::Backend(
                    "CUDA KNN reduction found fewer candidates than expected".to_string(),
                ));
            }
            let slot = packed as u32 as usize;
            let hit = out_hits.copy_to_vec_at(slot, 1)?[0];
            hits.push(hit);
            out_hits.copy_from_slice_at(
                slot,
                &[Hit {
                    id: u32::MAX,
                    dist_sq: hit.dist_sq,
                }],
            )?;
        }
        hits.sort_unstable_by(compare_hits);
        Ok(hits)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let _ = self.context.set_current();
    }
}

impl RingComputeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>) -> Result<()> {
        if dims == 0 {
            return Err(RingDbError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !vectors.len().is_multiple_of(dims) {
            return Err(RingDbError::Backend(format!(
                "CUDA upload got {} vector values, which is not divisible by {dims} dimensions",
                vectors.len()
            )));
        }
        let n_vectors = vectors.len() / dims;
        if n_vectors > u32::MAX as usize {
            return Err(RingDbError::Backend(
                "CUDA backend supports at most u32::MAX vectors".to_string(),
            ));
        }

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        self.dims = dims;
        self.n_vectors = n_vectors;
        self.vectors = Some(self.upload_transposed_vectors(&vectors, dims, self.n_vectors)?);
        self.out_hits = Some(DeviceBuffer::uninit(self.driver.clone(), self.n_vectors)?);
        self.out_count = Some(DeviceBuffer::from_slice(self.driver.clone(), &[0u32])?);
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_min: f32,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        self.validate_query(dims, query)?;
        if let Some(weights) = weights {
            Self::validate_weights(dims, weights)?;
            self.launch_weighted_ring(
                query,
                weights,
                d_min * d_min,
                d_max * d_max,
                false,
                candidates,
            )
        } else {
            self.launch_ring(query, d_min * d_min, d_max * d_max, false, candidates)
        }
    }

    fn disk_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        self.validate_query(dims, query)?;
        if let Some(weights) = weights {
            Self::validate_weights(dims, weights)?;
            self.launch_weighted_ring(query, weights, 0.0, d_max * d_max, true, candidates)
        } else {
            self.launch_ring(query, 0.0, d_max * d_max, true, candidates)
        }
    }

    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[DiskSpec<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        self.validate_disks(dims, disks)?;
        if disks.iter().all(|disk| disk.weights.is_none()) {
            self.launch_disk_intersection(disks, candidates)
        } else {
            self.launch_mixed_disk_intersection(disks, candidates)
        }
    }

    fn knn_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        self.validate_query(dims, query)?;
        if let Some(weights) = weights {
            Self::validate_weights(dims, weights)?;
        }
        self.launch_knn(query, weights, k, candidates)
    }
}

const CUDA_PTX: &str = concat!(
    r#"
.version 6.4
.target sm_52
.address_size 64

.visible .entry ring_query_f32_kernel(
    .param .u64 ring_vectors,
    .param .u32 ring_dims,
    .param .u32 ring_n_vectors,
    .param .u32 ring_candidate_mode,
    .param .u64 ring_candidates,
    .param .u32 ring_work_count,
    .param .u64 ring_query,
    .param .f32 ring_lower_sq,
    .param .f32 ring_upper_sq,
    .param .u64 ring_out_ids,
    .param .u64 ring_out_dists,
    .param .u64 ring_out_count
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [ring_vectors];
    ld.param.u32 %r5, [ring_dims];
    ld.param.u32 %r6, [ring_n_vectors];
    ld.param.u32 %r13, [ring_candidate_mode];
    ld.param.u64 %rd24, [ring_candidates];
    ld.param.u32 %r14, [ring_work_count];
    ld.param.u64 %rd3, [ring_query];
    ld.param.f32 %f8, [ring_lower_sq];
    ld.param.f32 %f9, [ring_upper_sq];
    ld.param.u64 %rd4, [ring_out_ids];
    ld.param.u64 %rd5, [ring_out_dists];
    ld.param.u64 %rd6, [ring_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p6, %r13, 0;
    @%p6 bra ring_candidate_ready;
    setp.eq.u32 %p7, %r13, 1;
    @%p7 bra ring_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd25, %r15, 8;
    add.u64 %rd26, %rd24, %rd25;
    ld.global.u64 %rd27, [%rd26];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd28, 1;
    shl.b64 %rd28, %rd28, %r16;
    and.b64 %rd29, %rd27, %rd28;
    setp.eq.u64 %p8, %rd29, 0;
    @%p8 ret;
    bra ring_candidate_ready;

ring_candidate_sparse:
    mul.wide.u32 %rd25, %r4, 4;
    add.u64 %rd26, %rd24, %rd25;
    ld.global.u32 %r4, [%rd26];

ring_candidate_ready:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd1, %rd10;
    cvt.u64.u32 %rd12, %r6;
    shl.b64 %rd12, %rd12, 2;
    mov.u64 %rd16, %rd3;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

ring_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra ring_done;
    ld.global.f32 %f2, [%rd10];
    ld.global.f32 %f3, [%rd16];
    sub.rn.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f5, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u32 %r7, %r7, 1;
    bra ring_loop;

ring_done:
    setp.ge.f32 %p3, %f1, %f8;
    setp.le.f32 %p4, %f1, %f9;
    and.pred %p5, %p3, %p4;
    @!%p5 ret;

    atom.global.add.u32 %r8, [%rd6], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd4, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f1;
    ret;
}

.visible .entry disk_query_f32_kernel(
    .param .u64 disk_vectors,
    .param .u32 disk_dims,
    .param .u32 disk_n_vectors,
    .param .u32 disk_candidate_mode,
    .param .u64 disk_candidates,
    .param .u32 disk_work_count,
    .param .u64 disk_query,
    .param .f32 disk_upper_sq,
    .param .u64 disk_out_ids,
    .param .u64 disk_out_dists,
    .param .u64 disk_out_count
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<11>;

    ld.param.u64 %rd1, [disk_vectors];
    ld.param.u32 %r5, [disk_dims];
    ld.param.u32 %r6, [disk_n_vectors];
    ld.param.u32 %r13, [disk_candidate_mode];
    ld.param.u64 %rd24, [disk_candidates];
    ld.param.u32 %r14, [disk_work_count];
    ld.param.u64 %rd3, [disk_query];
    ld.param.f32 %f9, [disk_upper_sq];
    ld.param.u64 %rd4, [disk_out_ids];
    ld.param.u64 %rd5, [disk_out_dists];
    ld.param.u64 %rd6, [disk_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p5, %r13, 0;
    @%p5 bra disk_candidate_ready;
    setp.eq.u32 %p6, %r13, 1;
    @%p6 bra disk_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd25, %r15, 8;
    add.u64 %rd26, %rd24, %rd25;
    ld.global.u64 %rd27, [%rd26];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd28, 1;
    shl.b64 %rd28, %rd28, %r16;
    and.b64 %rd29, %rd27, %rd28;
    setp.eq.u64 %p7, %rd29, 0;
    @%p7 ret;
    bra disk_candidate_ready;

disk_candidate_sparse:
    mul.wide.u32 %rd25, %r4, 4;
    add.u64 %rd26, %rd24, %rd25;
    ld.global.u32 %r4, [%rd26];

disk_candidate_ready:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd1, %rd10;
    cvt.u64.u32 %rd12, %r6;
    shl.b64 %rd12, %rd12, 2;
    mov.u64 %rd16, %rd3;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

disk_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra disk_done;
    ld.global.f32 %f2, [%rd10];
    ld.global.f32 %f3, [%rd16];
    sub.rn.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f5, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u32 %r7, %r7, 1;
    bra disk_loop;

disk_done:
    setp.le.f32 %p3, %f1, %f9;
    @!%p3 ret;

    atom.global.add.u32 %r8, [%rd6], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd4, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f1;
    ret;
}

.visible .entry weighted_ring_query_f32_kernel(
    .param .u64 wring_vectors,
    .param .u32 wring_dims,
    .param .u32 wring_n_vectors,
    .param .u32 wring_candidate_mode,
    .param .u64 wring_candidates,
    .param .u32 wring_work_count,
    .param .u64 wring_query,
    .param .u64 wring_weights,
    .param .f32 wring_lower_sq,
    .param .f32 wring_upper_sq,
    .param .u64 wring_out_ids,
    .param .u64 wring_out_dists,
    .param .u64 wring_out_count
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<36>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [wring_vectors];
    ld.param.u32 %r5, [wring_dims];
    ld.param.u32 %r6, [wring_n_vectors];
    ld.param.u32 %r13, [wring_candidate_mode];
    ld.param.u64 %rd28, [wring_candidates];
    ld.param.u32 %r14, [wring_work_count];
    ld.param.u64 %rd3, [wring_query];
    ld.param.u64 %rd4, [wring_weights];
    ld.param.f32 %f8, [wring_lower_sq];
    ld.param.f32 %f9, [wring_upper_sq];
    ld.param.u64 %rd5, [wring_out_ids];
    ld.param.u64 %rd6, [wring_out_dists];
    ld.param.u64 %rd7, [wring_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p6, %r13, 0;
    @%p6 bra wring_candidate_ready;
    setp.eq.u32 %p7, %r13, 1;
    @%p7 bra wring_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd29, %r15, 8;
    add.u64 %rd30, %rd28, %rd29;
    ld.global.u64 %rd31, [%rd30];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd32, 1;
    shl.b64 %rd32, %rd32, %r16;
    and.b64 %rd33, %rd31, %rd32;
    setp.eq.u64 %p8, %rd33, 0;
    @%p8 ret;
    bra wring_candidate_ready;

wring_candidate_sparse:
    mul.wide.u32 %rd29, %r4, 4;
    add.u64 %rd30, %rd28, %rd29;
    ld.global.u32 %r4, [%rd30];

wring_candidate_ready:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd1, %rd10;
    cvt.u64.u32 %rd12, %r6;
    shl.b64 %rd12, %rd12, 2;
    mov.u64 %rd16, %rd3;
    mov.u64 %rd17, %rd4;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

wring_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra wring_done;
    ld.global.f32 %f4, [%rd17];
    setp.ne.f32 %p9, %f4, 0f00000000;
    @%p9 ld.global.f32 %f2, [%rd10];
    @%p9 ld.global.f32 %f3, [%rd16];
    @%p9 sub.rn.f32 %f5, %f2, %f3;
    @%p9 mul.rn.f32 %f6, %f5, %f5;
    @%p9 fma.rn.f32 %f1, %f4, %f6, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u64 %rd17, %rd17, 4;
    add.u32 %r7, %r7, 1;
    bra wring_loop;

wring_done:
    setp.ge.f32 %p3, %f1, %f8;
    setp.le.f32 %p4, %f1, %f9;
    and.pred %p5, %p3, %p4;
    @!%p5 ret;

    atom.global.add.u32 %r8, [%rd7], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd5, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f1;
    ret;
}

.visible .entry weighted_disk_query_f32_kernel(
    .param .u64 wdisk_vectors,
    .param .u32 wdisk_dims,
    .param .u32 wdisk_n_vectors,
    .param .u32 wdisk_candidate_mode,
    .param .u64 wdisk_candidates,
    .param .u32 wdisk_work_count,
    .param .u64 wdisk_query,
    .param .u64 wdisk_weights,
    .param .f32 wdisk_upper_sq,
    .param .u64 wdisk_out_ids,
    .param .u64 wdisk_out_dists,
    .param .u64 wdisk_out_count
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<36>;
    .reg .f32 %f<11>;

    ld.param.u64 %rd1, [wdisk_vectors];
    ld.param.u32 %r5, [wdisk_dims];
    ld.param.u32 %r6, [wdisk_n_vectors];
    ld.param.u32 %r13, [wdisk_candidate_mode];
    ld.param.u64 %rd28, [wdisk_candidates];
    ld.param.u32 %r14, [wdisk_work_count];
    ld.param.u64 %rd3, [wdisk_query];
    ld.param.u64 %rd4, [wdisk_weights];
    ld.param.f32 %f9, [wdisk_upper_sq];
    ld.param.u64 %rd5, [wdisk_out_ids];
    ld.param.u64 %rd6, [wdisk_out_dists];
    ld.param.u64 %rd7, [wdisk_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p4, %r13, 0;
    @%p4 bra wdisk_candidate_ready;
    setp.eq.u32 %p5, %r13, 1;
    @%p5 bra wdisk_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd29, %r15, 8;
    add.u64 %rd30, %rd28, %rd29;
    ld.global.u64 %rd31, [%rd30];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd32, 1;
    shl.b64 %rd32, %rd32, %r16;
    and.b64 %rd33, %rd31, %rd32;
    setp.eq.u64 %p6, %rd33, 0;
    @%p6 ret;
    bra wdisk_candidate_ready;

wdisk_candidate_sparse:
    mul.wide.u32 %rd29, %r4, 4;
    add.u64 %rd30, %rd28, %rd29;
    ld.global.u32 %r4, [%rd30];

wdisk_candidate_ready:

    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd1, %rd10;
    cvt.u64.u32 %rd12, %r6;
    shl.b64 %rd12, %rd12, 2;
    mov.u64 %rd16, %rd3;
    mov.u64 %rd17, %rd4;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

wdisk_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra wdisk_done;
    ld.global.f32 %f4, [%rd17];
    setp.ne.f32 %p7, %f4, 0f00000000;
    @%p7 ld.global.f32 %f2, [%rd10];
    @%p7 ld.global.f32 %f3, [%rd16];
    @%p7 sub.rn.f32 %f5, %f2, %f3;
    @%p7 mul.rn.f32 %f6, %f5, %f5;
    @%p7 fma.rn.f32 %f1, %f4, %f6, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u64 %rd17, %rd17, 4;
    add.u32 %r7, %r7, 1;
    bra wdisk_loop;

wdisk_done:
    setp.le.f32 %p3, %f1, %f9;
    @!%p3 ret;

    atom.global.add.u32 %r8, [%rd7], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd5, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f1;
    ret;
}

.visible .entry disk_intersection_query_f32_kernel(
    .param .u64 isect_vectors,
    .param .u32 isect_dims,
    .param .u32 isect_n_vectors,
    .param .u32 isect_candidate_mode,
    .param .u64 isect_candidates,
    .param .u32 isect_work_count,
    .param .u64 isect_disk_queries,
    .param .u64 isect_disk_upper_sq,
    .param .u32 isect_n_disks,
    .param .u64 isect_out_ids,
    .param .u64 isect_out_dists,
    .param .u64 isect_out_count
)
{
    .reg .pred %p<11>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<42>;
    .reg .f32 %f<14>;

    ld.param.u64 %rd1, [isect_vectors];
    ld.param.u32 %r5, [isect_dims];
    ld.param.u32 %r6, [isect_n_vectors];
    ld.param.u32 %r13, [isect_candidate_mode];
    ld.param.u64 %rd34, [isect_candidates];
    ld.param.u32 %r14, [isect_work_count];
    ld.param.u64 %rd3, [isect_disk_queries];
    ld.param.u64 %rd5, [isect_disk_upper_sq];
    ld.param.u32 %r10, [isect_n_disks];
    ld.param.u64 %rd6, [isect_out_ids];
    ld.param.u64 %rd7, [isect_out_dists];
    ld.param.u64 %rd8, [isect_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p7, %r13, 0;
    @%p7 bra isect_candidate_ready;
    setp.eq.u32 %p8, %r13, 1;
    @%p8 bra isect_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd35, %r15, 8;
    add.u64 %rd36, %rd34, %rd35;
    ld.global.u64 %rd37, [%rd36];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd38, 1;
    shl.b64 %rd38, %rd38, %r16;
    and.b64 %rd39, %rd37, %rd38;
    setp.eq.u64 %p9, %rd39, 0;
    @%p9 ret;
    bra isect_candidate_ready;

isect_candidate_sparse:
    mul.wide.u32 %rd35, %r4, 4;
    add.u64 %rd36, %rd34, %rd35;
    ld.global.u32 %r4, [%rd36];

isect_candidate_ready:
    mov.u32 %r9, 0;
    mov.f32 %f10, 0f00000000;

isect_disk_loop:
    setp.ge.u32 %p2, %r9, %r10;
    @%p2 bra isect_emit;

    mul.wide.u32 %rd24, %r9, %r5;
    shl.b64 %rd26, %rd24, 2;
    add.u64 %rd27, %rd3, %rd26;
    cvt.u64.u32 %rd32, %r4;
    shl.b64 %rd32, %rd32, 2;
    add.u64 %rd32, %rd1, %rd32;
    cvt.u64.u32 %rd33, %r6;
    shl.b64 %rd33, %rd33, 2;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

isect_dim_loop:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra isect_disk_done;
    ld.global.f32 %f2, [%rd32];
    ld.global.f32 %f3, [%rd27];
    sub.rn.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f5, %f1;
    add.u64 %rd32, %rd32, %rd33;
    add.u64 %rd27, %rd27, 4;
    add.u32 %r7, %r7, 1;
    bra isect_dim_loop;

isect_disk_done:
    cvt.u64.u32 %rd28, %r9;
    shl.b64 %rd29, %rd28, 2;
    add.u64 %rd31, %rd5, %rd29;
    ld.global.f32 %f9, [%rd31];
    setp.gt.f32 %p4, %f1, %f9;
    @%p4 ret;
    setp.eq.u32 %p5, %r9, 0;
    @%p5 mov.f32 %f10, %f1;
    add.u32 %r9, %r9, 1;
    bra isect_disk_loop;

isect_emit:
    atom.global.add.u32 %r8, [%rd8], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd6, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f10;
    ret;
}

.visible .entry mixed_disk_intersection_query_f32_kernel(
    .param .u64 mixed_vectors,
    .param .u32 mixed_dims,
    .param .u32 mixed_n_vectors,
    .param .u32 mixed_candidate_mode,
    .param .u64 mixed_candidates,
    .param .u32 mixed_work_count,
    .param .u64 mixed_disk_queries,
    .param .u64 mixed_disk_weights,
    .param .u64 mixed_disk_upper_sq,
    .param .u64 mixed_weight_offsets,
    .param .u32 mixed_n_disks,
    .param .u64 mixed_out_ids,
    .param .u64 mixed_out_dists,
    .param .u64 mixed_out_count
)
{
    .reg .pred %p<13>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<52>;
    .reg .f32 %f<14>;

    ld.param.u64 %rd1, [mixed_vectors];
    ld.param.u32 %r5, [mixed_dims];
    ld.param.u32 %r6, [mixed_n_vectors];
    ld.param.u32 %r13, [mixed_candidate_mode];
    ld.param.u64 %rd44, [mixed_candidates];
    ld.param.u32 %r14, [mixed_work_count];
    ld.param.u64 %rd3, [mixed_disk_queries];
    ld.param.u64 %rd4, [mixed_disk_weights];
    ld.param.u64 %rd6, [mixed_disk_upper_sq];
    ld.param.u64 %rd7, [mixed_weight_offsets];
    ld.param.u32 %r10, [mixed_n_disks];
    ld.param.u64 %rd8, [mixed_out_ids];
    ld.param.u64 %rd9, [mixed_out_dists];
    ld.param.u64 %rd10, [mixed_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;

    setp.eq.u32 %p9, %r13, 0;
    @%p9 bra mixed_candidate_ready;
    setp.eq.u32 %p10, %r13, 1;
    @%p10 bra mixed_candidate_sparse;
    shr.u32 %r15, %r4, 6;
    mul.wide.u32 %rd45, %r15, 8;
    add.u64 %rd46, %rd44, %rd45;
    ld.global.u64 %rd47, [%rd46];
    and.b32 %r16, %r4, 63;
    mov.u64 %rd48, 1;
    shl.b64 %rd48, %rd48, %r16;
    and.b64 %rd49, %rd47, %rd48;
    setp.eq.u64 %p11, %rd49, 0;
    @%p11 ret;
    bra mixed_candidate_ready;

mixed_candidate_sparse:
    mul.wide.u32 %rd45, %r4, 4;
    add.u64 %rd46, %rd44, %rd45;
    ld.global.u32 %r4, [%rd46];

mixed_candidate_ready:
    mov.u32 %r9, 0;
    mov.f32 %f10, 0f00000000;

mixed_disk_loop:
    setp.ge.u32 %p2, %r9, %r10;
    @%p2 bra mixed_emit;

    mul.wide.u32 %rd24, %r9, %r5;
    cvt.u64.u32 %rd25, %r9;
    shl.b64 %rd26, %rd25, 2;
    add.u64 %rd27, %rd7, %rd26;
    ld.global.u32 %r11, [%rd27];
    shl.b64 %rd29, %rd24, 2;
    add.u64 %rd30, %rd3, %rd29;

    cvt.u64.u32 %rd32, %r4;
    shl.b64 %rd32, %rd32, 2;
    add.u64 %rd32, %rd1, %rd32;
    cvt.u64.u32 %rd33, %r6;
    shl.b64 %rd33, %rd33, 2;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

    setp.eq.u32 %p6, %r11, 0xffffffff;
    @%p6 bra mixed_unweighted_loop;
    cvt.u64.u32 %rd34, %r11;
    shl.b64 %rd36, %rd34, 2;
    add.u64 %rd37, %rd4, %rd36;

mixed_weighted_loop:
    setp.ge.u32 %p4, %r7, %r5;
    @%p4 bra mixed_weighted_done;
    ld.global.f32 %f7, [%rd37];
    setp.ne.f32 %p12, %f7, 0f00000000;
    @%p12 ld.global.f32 %f2, [%rd32];
    @%p12 ld.global.f32 %f3, [%rd30];
    @%p12 sub.rn.f32 %f5, %f2, %f3;
    @%p12 mul.rn.f32 %f8, %f5, %f5;
    @%p12 fma.rn.f32 %f1, %f7, %f8, %f1;
    add.u64 %rd32, %rd32, %rd33;
    add.u64 %rd30, %rd30, 4;
    add.u64 %rd37, %rd37, 4;
    add.u32 %r7, %r7, 1;
    bra mixed_weighted_loop;

mixed_weighted_done:
    mov.f32 %f6, %f1;
    bra mixed_compare;

mixed_unweighted_loop:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra mixed_unweighted_done;
    ld.global.f32 %f2, [%rd32];
    ld.global.f32 %f3, [%rd30];
    sub.rn.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f5, %f1;
    add.u64 %rd32, %rd32, %rd33;
    add.u64 %rd30, %rd30, 4;
    add.u32 %r7, %r7, 1;
    bra mixed_unweighted_loop;

mixed_unweighted_done:
    mov.f32 %f6, %f1;

mixed_compare:
    add.u64 %rd31, %rd6, %rd26;
    ld.global.f32 %f9, [%rd31];
    setp.gt.f32 %p5, %f6, %f9;
    @%p5 ret;
    setp.eq.u32 %p7, %r9, 0;
    @%p7 mov.f32 %f10, %f6;
    add.u32 %r9, %r9, 1;
    bra mixed_disk_loop;

mixed_emit:
    atom.global.add.u32 %r8, [%rd10], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 3;
    add.u64 %rd22, %rd8, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd22, 4;
    st.global.f32 [%rd23], %f10;
    ret;
}

.visible .entry disk_intersection3_query_f32_kernel(
    .param .u64 isect3_vectors,
    .param .u32 isect3_dims,
    .param .u32 isect3_n_vectors,
    .param .u32 isect3_candidate_mode,
    .param .u64 isect3_candidates,
    .param .u32 isect3_work_count,
    .param .u64 isect3_disk_queries,
    .param .u64 isect3_disk_upper_sq,
    .param .u32 isect3_n_disks,
    .param .u64 isect3_out_ids,
    .param .u64 isect3_out_dists,
    .param .u64 isect3_out_count
)
{
    .reg .pred %p<14>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<60>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [isect3_vectors];
    ld.param.u32 %r5, [isect3_dims];
    ld.param.u32 %r6, [isect3_n_vectors];
    ld.param.u32 %r33, [isect3_candidate_mode];
    ld.param.u64 %rd52, [isect3_candidates];
    ld.param.u32 %r34, [isect3_work_count];
    ld.param.u64 %rd3, [isect3_disk_queries];
    ld.param.u64 %rd5, [isect3_disk_upper_sq];
    ld.param.u32 %r10, [isect3_n_disks];
    ld.param.u64 %rd6, [isect3_out_ids];
    ld.param.u64 %rd7, [isect3_out_dists];
    ld.param.u64 %rd8, [isect3_out_count];

    setp.ne.u32 %p1, %r10, 3;
    @%p1 ret;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    and.b32 %r11, %r3, 31;
    shr.u32 %r12, %r3, 5;
    setp.ge.u32 %p2, %r4, %r34;
    @%p2 ret;

    setp.eq.u32 %p10, %r33, 0;
    @%p10 bra isect3_candidate_ready;
    setp.eq.u32 %p11, %r33, 1;
    @%p11 bra isect3_candidate_sparse;
    shr.u32 %r35, %r4, 6;
    mul.wide.u32 %rd53, %r35, 8;
    add.u64 %rd54, %rd52, %rd53;
    ld.global.u64 %rd55, [%rd54];
    and.b32 %r36, %r4, 63;
    mov.u64 %rd56, 1;
    shl.b64 %rd56, %rd56, %r36;
    and.b64 %rd57, %rd55, %rd56;
    setp.eq.u64 %p12, %rd57, 0;
    @%p12 ret;
    bra isect3_candidate_ready;

isect3_candidate_sparse:
    mul.wide.u32 %rd53, %r4, 4;
    add.u64 %rd54, %rd52, %rd53;
    ld.global.u32 %r4, [%rd54];

isect3_candidate_ready:

    cvt.u64.u32 %rd44, %r4;
    shl.b64 %rd44, %rd44, 2;
    add.u64 %rd44, %rd1, %rd44;
    cvt.u64.u32 %rd45, %r6;
    shl.b64 %rd45, %rd45, 2;
    cvt.u64.u32 %rd46, %r5;
    shl.b64 %rd46, %rd46, 2;
    mov.u64 %rd47, %rd3;
    add.u64 %rd48, %rd3, %rd46;
    add.u64 %rd49, %rd48, %rd46;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

isect3_dim_loop:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra isect3_done;
    ld.global.f32 %f4, [%rd44];
    ld.global.f32 %f5, [%rd47];
    sub.rn.f32 %f8, %f4, %f5;
    fma.rn.f32 %f1, %f8, %f8, %f1;
    ld.global.f32 %f6, [%rd48];
    sub.rn.f32 %f9, %f4, %f6;
    fma.rn.f32 %f2, %f9, %f9, %f2;
    ld.global.f32 %f7, [%rd49];
    sub.rn.f32 %f10, %f4, %f7;
    fma.rn.f32 %f3, %f10, %f10, %f3;
    add.u64 %rd44, %rd44, %rd45;
    add.u64 %rd47, %rd47, 4;
    add.u64 %rd48, %rd48, 4;
    add.u64 %rd49, %rd49, 4;
    add.u32 %r7, %r7, 1;
    bra isect3_dim_loop;

isect3_done:
    ld.global.f32 %f15, [%rd5];
    add.u64 %rd30, %rd5, 4;
    ld.global.f32 %f16, [%rd30];
    add.u64 %rd31, %rd5, 8;
    ld.global.f32 %f17, [%rd31];

    setp.le.f32 %p5, %f1, %f15;
    setp.le.f32 %p6, %f2, %f16;
    and.pred %p5, %p5, %p6;
    setp.le.f32 %p7, %f3, %f17;
    and.pred %p5, %p5, %p7;

    vote.ballot.b32 %r21, %p5;
    setp.eq.u32 %p8, %r21, 0;
    @%p8 ret;
    mov.u32 %r22, %lanemask_lt;
    and.b32 %r23, %r21, %r22;
    popc.b32 %r24, %r23;
    popc.b32 %r25, %r21;

    setp.ne.u32 %p8, %r11, 0;
    @%p8 bra isect3_skip_warp_reserve;
    atom.global.add.u32 %r31, [%rd8], %r25;

isect3_skip_warp_reserve:
    shfl.idx.b32 %r31, %r31, 0, 31;
    @!%p5 ret;

    add.u32 %r32, %r31, %r24;
    cvt.u64.u32 %rd40, %r32;
    shl.b64 %rd41, %rd40, 3;
    add.u64 %rd42, %rd6, %rd41;
    st.global.u32 [%rd42], %r4;
    add.u64 %rd43, %rd42, 4;
    st.global.f32 [%rd43], %f1;
    ret;
}

.visible .entry knn_distance_f32_kernel(
    .param .u64 knn_vectors,
    .param .u32 knn_dims,
    .param .u32 knn_n_vectors,
    .param .u32 knn_candidate_mode,
    .param .u64 knn_candidates,
    .param .u32 knn_work_count,
    .param .u64 knn_query,
    .param .u64 knn_weights,
    .param .u32 knn_weighted,
    .param .u64 knn_out_hits
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [knn_vectors];
    ld.param.u32 %r5, [knn_dims];
    ld.param.u32 %r6, [knn_n_vectors];
    ld.param.u32 %r13, [knn_candidate_mode];
    ld.param.u64 %rd3, [knn_candidates];
    ld.param.u32 %r14, [knn_work_count];
    ld.param.u64 %rd4, [knn_query];
    ld.param.u64 %rd5, [knn_weights];
    ld.param.u32 %r15, [knn_weighted];
    ld.param.u64 %rd6, [knn_out_hits];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r14;
    @%p1 ret;
    mov.u32 %r12, %r4;

    setp.eq.u32 %p2, %r13, 0;
    @%p2 bra knn_candidate_ready;
    setp.eq.u32 %p3, %r13, 1;
    @%p3 bra knn_candidate_sparse;
    shr.u32 %r16, %r4, 6;
    mul.wide.u32 %rd20, %r16, 8;
    add.u64 %rd21, %rd3, %rd20;
    ld.global.u64 %rd22, [%rd21];
    and.b32 %r17, %r4, 63;
    mov.u64 %rd23, 1;
    shl.b64 %rd23, %rd23, %r17;
    and.b64 %rd24, %rd22, %rd23;
    setp.eq.u64 %p4, %rd24, 0;
    @%p4 bra knn_ineligible;
    bra knn_candidate_ready;

knn_candidate_sparse:
    mul.wide.u32 %rd20, %r4, 4;
    add.u64 %rd21, %rd3, %rd20;
    ld.global.u32 %r4, [%rd21];

knn_candidate_ready:
    cvt.u64.u32 %rd10, %r4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd1, %rd10;
    cvt.u64.u32 %rd12, %r6;
    shl.b64 %rd12, %rd12, 2;
    setp.eq.u32 %p5, %r15, 0;
    @%p5 bra knn_unweighted;

    mov.u64 %rd16, %rd4;
    mov.u64 %rd17, %rd5;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

knn_weighted_loop:
    setp.ge.u32 %p6, %r7, %r5;
    @%p6 bra knn_weighted_done;
    ld.global.f32 %f4, [%rd17];
    setp.ne.f32 %p8, %f4, 0f00000000;
    @%p8 ld.global.f32 %f2, [%rd10];
    @%p8 ld.global.f32 %f3, [%rd16];
    @%p8 sub.rn.f32 %f5, %f2, %f3;
    @%p8 mul.rn.f32 %f6, %f5, %f5;
    @%p8 fma.rn.f32 %f1, %f4, %f6, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u64 %rd17, %rd17, 4;
    add.u32 %r7, %r7, 1;
    bra knn_weighted_loop;

knn_weighted_done:
    mov.f32 %f10, %f1;
    bra knn_emit;

knn_unweighted:
    mov.u64 %rd16, %rd4;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

knn_unweighted_loop:
    setp.ge.u32 %p7, %r7, %r5;
    @%p7 bra knn_unweighted_done;
    ld.global.f32 %f2, [%rd10];
    ld.global.f32 %f3, [%rd16];
    sub.rn.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f5, %f1;
    add.u64 %rd10, %rd10, %rd12;
    add.u64 %rd16, %rd16, 4;
    add.u32 %r7, %r7, 1;
    bra knn_unweighted_loop;

knn_unweighted_done:
    mov.f32 %f10, %f1;

knn_emit:
    cvt.u64.u32 %rd25, %r12;
    shl.b64 %rd25, %rd25, 3;
    add.u64 %rd26, %rd6, %rd25;
    st.global.u32 [%rd26], %r4;
    add.u64 %rd27, %rd26, 4;
    st.global.f32 [%rd27], %f10;
    ret;

knn_ineligible:
    cvt.u64.u32 %rd25, %r12;
    shl.b64 %rd25, %rd25, 3;
    add.u64 %rd26, %rd6, %rd25;
    mov.u32 %r18, 0xffffffff;
    st.global.u32 [%rd26], %r18;
    ret;
}

.visible .entry knn_argmin_f32_kernel(
    .param .u64 argmin_hits,
    .param .u32 argmin_work_count,
    .param .u64 argmin_best
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [argmin_hits];
    ld.param.u32 %r5, [argmin_work_count];
    ld.param.u64 %rd2, [argmin_best];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 ret;

    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 3;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r6, [%rd5];
    setp.eq.u32 %p2, %r6, 0xffffffff;
    @%p2 ret;

    add.u64 %rd6, %rd5, 4;
    ld.global.u32 %r7, [%rd6];
    shr.u32 %r8, %r7, 31;
    neg.s32 %r9, %r8;
    or.b32 %r9, %r9, 0x80000000;
    xor.b32 %r10, %r7, %r9;
    cvt.u64.u32 %rd7, %r10;
    shl.b64 %rd7, %rd7, 32;
    or.b64 %rd8, %rd7, %rd3;
    atom.global.min.u64 %rd9, [%rd2], %rd8;
    ret;
}
"#,
    "\0"
);

#[cfg(test)]
mod tests {
    use super::*;

    fn ptx_section(start: &str, end: &str) -> &'static str {
        let (_, after_start) = CUDA_PTX.split_once(start).unwrap();
        let (section, _) = after_start.split_once(end).unwrap();
        section
    }

    fn assert_zero_weight_guard(section: &str, guard: &str, square: &str, accumulate: &str) {
        let guard_pos = section.find(guard).unwrap();
        let square_pos = section.find(square).unwrap();
        let accumulate_pos = section.find(accumulate).unwrap();
        assert!(guard_pos < square_pos);
        assert!(square_pos < accumulate_pos);
        assert!(
            !section
                .lines()
                .any(|line| line.trim_start().starts_with("mul.rn.f32"))
        );
    }

    fn assert_direct_delta(section: &str, delta: &str, accumulate: &str) {
        let delta_pos = section.find(delta).unwrap();
        let accumulate_pos = section.find(accumulate).unwrap();
        assert!(delta_pos < accumulate_pos);
    }

    #[test]
    fn dense_count_ignores_bits_outside_the_universe() {
        assert_eq!(dense_candidate_count(&[u64::MAX], 3), 3);
        assert_eq!(dense_candidate_count(&[u64::MAX, u64::MAX], 65), 65);
        assert_eq!(dense_candidate_count(&[0b1010], 4), 2);
    }

    #[test]
    fn knn_selection_uses_the_actual_transfer_size() {
        assert!(select_knn_on_host(1, HOST_SELECT_RATIO));
        assert!(!select_knn_on_host(1, HOST_SELECT_RATIO + 1));
        assert!(select_knn_on_host(65, 65 * HOST_SELECT_RATIO));
    }

    #[test]
    fn hit_order_uses_id_to_break_distance_ties() {
        let mut hits = [
            Hit {
                id: 7,
                dist_sq: 1.0,
            },
            Hit {
                id: 2,
                dist_sq: 1.0,
            },
            Hit {
                id: 9,
                dist_sq: -0.0,
            },
        ];
        hits.sort_unstable_by(compare_hits);
        assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [9, 2, 7]);
    }

    #[test]
    fn unweighted_ptx_uses_direct_squared_differences_without_cached_norms() {
        assert!(!CUDA_PTX.contains("norm"));
        assert!(!CUDA_PTX.contains("0fC0000000"));

        for section in [
            ptx_section("\nring_loop:", "\nring_done:"),
            ptx_section("\ndisk_loop:", "\ndisk_done:"),
            ptx_section("\nisect_dim_loop:", "\nisect_disk_done:"),
            ptx_section("\nmixed_unweighted_loop:", "\nmixed_unweighted_done:"),
            ptx_section("\nknn_unweighted_loop:", "\nknn_unweighted_done:"),
        ] {
            assert_direct_delta(
                section,
                "sub.rn.f32 %f5, %f2, %f3;",
                "fma.rn.f32 %f1, %f5, %f5, %f1;",
            );
        }

        let intersection3 = ptx_section("\nisect3_dim_loop:", "\nisect3_done:");
        for (delta, accumulate) in [
            (
                "sub.rn.f32 %f8, %f4, %f5;",
                "fma.rn.f32 %f1, %f8, %f8, %f1;",
            ),
            (
                "sub.rn.f32 %f9, %f4, %f6;",
                "fma.rn.f32 %f2, %f9, %f9, %f2;",
            ),
            (
                "sub.rn.f32 %f10, %f4, %f7;",
                "fma.rn.f32 %f3, %f10, %f10, %f3;",
            ),
        ] {
            assert_direct_delta(intersection3, delta, accumulate);
        }
    }

    #[test]
    fn weighted_ptx_skips_zero_weight_dimensions_before_squaring() {
        let loops = [
            (
                ptx_section("wring_loop:", "wring_done:"),
                "setp.ne.f32 %p9, %f4, 0f00000000;",
                "@%p9 mul.rn.f32 %f6, %f5, %f5;",
                "@%p9 fma.rn.f32 %f1, %f4, %f6, %f1;",
            ),
            (
                ptx_section("wdisk_loop:", "wdisk_done:"),
                "setp.ne.f32 %p7, %f4, 0f00000000;",
                "@%p7 mul.rn.f32 %f6, %f5, %f5;",
                "@%p7 fma.rn.f32 %f1, %f4, %f6, %f1;",
            ),
            (
                ptx_section("mixed_weighted_loop:", "mixed_weighted_done:"),
                "setp.ne.f32 %p12, %f7, 0f00000000;",
                "@%p12 mul.rn.f32 %f8, %f5, %f5;",
                "@%p12 fma.rn.f32 %f1, %f7, %f8, %f1;",
            ),
            (
                ptx_section("knn_weighted_loop:", "knn_weighted_done:"),
                "setp.ne.f32 %p8, %f4, 0f00000000;",
                "@%p8 mul.rn.f32 %f6, %f5, %f5;",
                "@%p8 fma.rn.f32 %f1, %f4, %f6, %f1;",
            ),
        ];

        for (section, guard, square, accumulate) in loops {
            assert_zero_weight_guard(section, guard, square, accumulate);
        }
    }
}
