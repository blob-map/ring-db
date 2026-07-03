use crate::backend::{QueryResponse, RingComputeBackend};
use crate::error::{Result, RingDbError};
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

    fn copy_to_vec(&self, len: usize) -> Result<Vec<T>> {
        assert!(len <= self.len);
        let mut out: Vec<T> = Vec::with_capacity(len);
        if len > 0 {
            unsafe {
                self.driver.check(
                    (self.driver.cu_memcpy_dtoh)(
                        out.as_mut_ptr().cast(),
                        self.ptr,
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
/// launches one CUDA thread per vector, computes the dot product on-device, and
/// compacts hits into output buffers via a device-side atomic counter.
pub struct CudaBackend {
    vectors: Option<DeviceBuffer<f32>>,
    norms_sq: Option<DeviceBuffer<f32>>,
    out_ids: Option<DeviceBuffer<u32>>,
    out_dists: Option<DeviceBuffer<f32>>,
    out_count: Option<DeviceBuffer<u32>>,
    _module: CudaModule,
    context: CudaContext,
    driver: Arc<CudaDriver>,
    launch_lock: Mutex<()>,
    dims: usize,
    n_vectors: usize,
    ring_kernel: CUfunction,
    disk_kernel: CUfunction,
    disk_intersection_kernel: CUfunction,
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
        let disk_intersection_kernel = module.function(b"disk_intersection_query_f32_kernel\0")?;

        Ok(Self {
            vectors: None,
            norms_sq: None,
            out_ids: None,
            out_dists: None,
            out_count: None,
            _module: module,
            context,
            driver,
            launch_lock: Mutex::new(()),
            dims: 0,
            n_vectors: 0,
            ring_kernel,
            disk_kernel,
            disk_intersection_kernel,
        })
    }

    fn query_norm_sq(query: &[f32]) -> f32 {
        query.iter().map(|x| x * x).sum()
    }

    fn require_uploaded(&self) -> Result<(&DeviceBuffer<f32>, &DeviceBuffer<f32>)> {
        let vectors = self.vectors.as_ref().ok_or_else(|| {
            RingDbError::Backend("CUDA dataset has not been uploaded".to_string())
        })?;
        let norms_sq = self
            .norms_sq
            .as_ref()
            .ok_or_else(|| RingDbError::Backend("CUDA norms have not been uploaded".to_string()))?;
        Ok((vectors, norms_sq))
    }

    fn require_output_buffers(
        &self,
    ) -> Result<(&DeviceBuffer<u32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>)> {
        let out_ids = self
            .out_ids
            .as_ref()
            .ok_or_else(|| RingDbError::Backend("CUDA output IDs are not allocated".to_string()))?;
        let out_dists = self.out_dists.as_ref().ok_or_else(|| {
            RingDbError::Backend("CUDA output distances are not allocated".to_string())
        })?;
        let out_count = self.out_count.as_ref().ok_or_else(|| {
            RingDbError::Backend("CUDA output counter is not allocated".to_string())
        })?;
        Ok((out_ids, out_dists, out_count))
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

    fn launch_ring(
        &self,
        query: &[f32],
        lower_sq: f32,
        upper_sq: f32,
        disk_mode: bool,
    ) -> Result<Vec<QueryResponse>> {
        if self.n_vectors == 0 {
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

        let (vectors, norms_sq) = self.require_uploaded()?;
        let (out_ids, out_dists, out_count) = self.require_output_buffers()?;
        let query_buf = DeviceBuffer::from_slice(self.driver.clone(), query)?;
        let zero = [0u32];
        out_count.copy_from_slice(&zero)?;

        let mut vectors_ptr = vectors.ptr;
        let mut norms_ptr = norms_sq.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut query_ptr = query_buf.ptr;
        let mut norm_sq_q = Self::query_norm_sq(query);
        let mut lower_sq = lower_sq;
        let mut upper_sq = upper_sq;
        let mut out_ids_ptr = out_ids.ptr;
        let mut out_dists_ptr = out_dists.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = n_vectors_u32.div_ceil(THREADS_PER_BLOCK);
        let mut ring_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut norms_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut norm_sq_q as *mut f32).cast::<c_void>(),
            (&mut lower_sq as *mut f32).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_ids_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_dists_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];
        let mut disk_params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut norms_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut query_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut norm_sq_q as *mut f32).cast::<c_void>(),
            (&mut upper_sq as *mut f32).cast::<c_void>(),
            (&mut out_ids_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_dists_ptr as *mut CUdeviceptr).cast::<c_void>(),
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
        let hit_count = hit_count.min(self.n_vectors);
        let ids = out_ids.copy_to_vec(hit_count)?;
        let dists = out_dists.copy_to_vec(hit_count)?;

        Ok(ids
            .into_iter()
            .zip(dists)
            .map(|(id, dist_sq)| QueryResponse { id, dist_sq })
            .collect())
    }

    fn validate_disks(&self, dims: usize, disks: &[(&[f32], f32)]) -> Result<()> {
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
        for &(query, _) in disks {
            if query.len() != dims {
                return Err(RingDbError::DimensionMismatch {
                    expected: dims,
                    got: query.len(),
                });
            }
        }
        Ok(())
    }

    fn launch_disk_intersection(&self, disks: &[(&[f32], f32)]) -> Result<Vec<QueryResponse>> {
        if self.n_vectors == 0 || disks.is_empty() {
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
        let mut disk_norms_sq = Vec::with_capacity(disks.len());
        let mut disk_upper_sq = Vec::with_capacity(disks.len());
        for &(query, d_max) in disks {
            disk_queries.extend_from_slice(query);
            disk_norms_sq.push(Self::query_norm_sq(query));
            disk_upper_sq.push(d_max * d_max);
        }

        let _guard = self
            .launch_lock
            .lock()
            .map_err(|_| RingDbError::Backend("CUDA launch mutex is poisoned".to_string()))?;
        self.context.set_current()?;

        let (vectors, norms_sq) = self.require_uploaded()?;
        let (out_ids, out_dists, out_count) = self.require_output_buffers()?;
        let disk_queries_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_queries)?;
        let disk_norms_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_norms_sq)?;
        let disk_upper_buf = DeviceBuffer::from_slice(self.driver.clone(), &disk_upper_sq)?;
        out_count.copy_from_slice(&[0u32])?;

        let mut vectors_ptr = vectors.ptr;
        let mut norms_ptr = norms_sq.ptr;
        let mut dims_u32 = self.dims as u32;
        let mut n_vectors_u32 = self.n_vectors as u32;
        let mut disk_queries_ptr = disk_queries_buf.ptr;
        let mut disk_norms_ptr = disk_norms_buf.ptr;
        let mut disk_upper_ptr = disk_upper_buf.ptr;
        let mut n_disks_u32 = disks.len() as u32;
        let mut out_ids_ptr = out_ids.ptr;
        let mut out_dists_ptr = out_dists.ptr;
        let mut out_count_ptr = out_count.ptr;

        let grid_x = n_vectors_u32.div_ceil(THREADS_PER_BLOCK);
        let mut params = [
            (&mut vectors_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut norms_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut dims_u32 as *mut u32).cast::<c_void>(),
            (&mut n_vectors_u32 as *mut u32).cast::<c_void>(),
            (&mut disk_queries_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut disk_norms_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut disk_upper_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut n_disks_u32 as *mut u32).cast::<c_void>(),
            (&mut out_ids_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_dists_ptr as *mut CUdeviceptr).cast::<c_void>(),
            (&mut out_count_ptr as *mut CUdeviceptr).cast::<c_void>(),
        ];

        unsafe {
            self.driver.check(
                (self.driver.cu_launch_kernel)(
                    self.disk_intersection_kernel,
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
        let hit_count = hit_count.min(self.n_vectors);
        let ids = out_ids.copy_to_vec(hit_count)?;
        let dists = out_dists.copy_to_vec(hit_count)?;

        Ok(ids
            .into_iter()
            .zip(dists)
            .map(|(id, dist_sq)| QueryResponse { id, dist_sq })
            .collect())
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

    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()> {
        if dims == 0 {
            return Err(RingDbError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let expected_values = dims.checked_mul(norms_sq.len()).ok_or_else(|| {
            RingDbError::Backend("CUDA upload vector length overflow".to_string())
        })?;
        if vectors.len() != expected_values {
            return Err(RingDbError::Backend(format!(
                "CUDA upload got {} vector values, expected {}",
                vectors.len(),
                expected_values
            )));
        }
        if norms_sq.len() > u32::MAX as usize {
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
        self.n_vectors = norms_sq.len();
        self.vectors = Some(DeviceBuffer::from_slice(self.driver.clone(), &vectors)?);
        self.norms_sq = Some(DeviceBuffer::from_slice(self.driver.clone(), &norms_sq)?);
        self.out_ids = Some(DeviceBuffer::uninit(self.driver.clone(), self.n_vectors)?);
        self.out_dists = Some(DeviceBuffer::uninit(self.driver.clone(), self.n_vectors)?);
        self.out_count = Some(DeviceBuffer::from_slice(self.driver.clone(), &[0u32])?);
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Result<Vec<QueryResponse>> {
        self.validate_query(dims, query)?;
        self.launch_ring(query, d_min * d_min, d_max * d_max, false)
    }

    fn disk_query_f32(&self, dims: usize, query: &[f32], d_max: f32) -> Result<Vec<QueryResponse>> {
        self.validate_query(dims, query)?;
        self.launch_ring(query, 0.0, d_max * d_max, true)
    }

    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[(&[f32], f32)],
    ) -> Result<Vec<QueryResponse>> {
        self.validate_disks(dims, disks)?;
        self.launch_disk_intersection(disks)
    }
}

const CUDA_PTX: &str = concat!(
    r#"
.version 6.4
.target sm_52
.address_size 64

.visible .entry ring_query_f32_kernel(
    .param .u64 ring_vectors,
    .param .u64 ring_norms_sq,
    .param .u32 ring_dims,
    .param .u32 ring_n_vectors,
    .param .u64 ring_query,
    .param .f32 ring_norm_sq_q,
    .param .f32 ring_lower_sq,
    .param .f32 ring_upper_sq,
    .param .u64 ring_out_ids,
    .param .u64 ring_out_dists,
    .param .u64 ring_out_count
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [ring_vectors];
    ld.param.u64 %rd2, [ring_norms_sq];
    ld.param.u32 %r5, [ring_dims];
    ld.param.u32 %r6, [ring_n_vectors];
    ld.param.u64 %rd3, [ring_query];
    ld.param.f32 %f7, [ring_norm_sq_q];
    ld.param.f32 %f8, [ring_lower_sq];
    ld.param.f32 %f9, [ring_upper_sq];
    ld.param.u64 %rd4, [ring_out_ids];
    ld.param.u64 %rd5, [ring_out_dists];
    ld.param.u64 %rd6, [ring_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r6;
    @%p1 ret;

    mul.wide.u32 %rd10, %r4, %r5;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

ring_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra ring_done;
    cvt.u64.u32 %rd11, %r7;
    add.u64 %rd12, %rd10, %rd11;
    shl.b64 %rd13, %rd12, 2;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.f32 %f2, [%rd14];
    shl.b64 %rd15, %rd11, 2;
    add.u64 %rd16, %rd3, %rd15;
    ld.global.f32 %f3, [%rd16];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra ring_loop;

ring_done:
    cvt.u64.u32 %rd17, %r4;
    shl.b64 %rd18, %rd17, 2;
    add.u64 %rd19, %rd2, %rd18;
    ld.global.f32 %f4, [%rd19];
    fma.rn.f32 %f5, %f1, 0fC0000000, %f4;
    add.rn.f32 %f6, %f5, %f7;
    setp.ge.f32 %p3, %f6, %f8;
    setp.le.f32 %p4, %f6, %f9;
    and.pred %p5, %p3, %p4;
    @!%p5 ret;

    atom.global.add.u32 %r8, [%rd6], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 2;
    add.u64 %rd22, %rd4, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd5, %rd21;
    st.global.f32 [%rd23], %f6;
    ret;
}

.visible .entry disk_query_f32_kernel(
    .param .u64 disk_vectors,
    .param .u64 disk_norms_sq,
    .param .u32 disk_dims,
    .param .u32 disk_n_vectors,
    .param .u64 disk_query,
    .param .f32 disk_norm_sq_q,
    .param .f32 disk_upper_sq,
    .param .u64 disk_out_ids,
    .param .u64 disk_out_dists,
    .param .u64 disk_out_count
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<11>;

    ld.param.u64 %rd1, [disk_vectors];
    ld.param.u64 %rd2, [disk_norms_sq];
    ld.param.u32 %r5, [disk_dims];
    ld.param.u32 %r6, [disk_n_vectors];
    ld.param.u64 %rd3, [disk_query];
    ld.param.f32 %f7, [disk_norm_sq_q];
    ld.param.f32 %f9, [disk_upper_sq];
    ld.param.u64 %rd4, [disk_out_ids];
    ld.param.u64 %rd5, [disk_out_dists];
    ld.param.u64 %rd6, [disk_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r6;
    @%p1 ret;

    mul.wide.u32 %rd10, %r4, %r5;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

disk_loop:
    setp.ge.u32 %p2, %r7, %r5;
    @%p2 bra disk_done;
    cvt.u64.u32 %rd11, %r7;
    add.u64 %rd12, %rd10, %rd11;
    shl.b64 %rd13, %rd12, 2;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.f32 %f2, [%rd14];
    shl.b64 %rd15, %rd11, 2;
    add.u64 %rd16, %rd3, %rd15;
    ld.global.f32 %f3, [%rd16];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra disk_loop;

disk_done:
    cvt.u64.u32 %rd17, %r4;
    shl.b64 %rd18, %rd17, 2;
    add.u64 %rd19, %rd2, %rd18;
    ld.global.f32 %f4, [%rd19];
    fma.rn.f32 %f5, %f1, 0fC0000000, %f4;
    add.rn.f32 %f6, %f5, %f7;
    setp.le.f32 %p3, %f6, %f9;
    @!%p3 ret;

    atom.global.add.u32 %r8, [%rd6], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 2;
    add.u64 %rd22, %rd4, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd5, %rd21;
    st.global.f32 [%rd23], %f6;
    ret;
}

.visible .entry disk_intersection_query_f32_kernel(
    .param .u64 isect_vectors,
    .param .u64 isect_norms_sq,
    .param .u32 isect_dims,
    .param .u32 isect_n_vectors,
    .param .u64 isect_disk_queries,
    .param .u64 isect_disk_norms_sq,
    .param .u64 isect_disk_upper_sq,
    .param .u32 isect_n_disks,
    .param .u64 isect_out_ids,
    .param .u64 isect_out_dists,
    .param .u64 isect_out_count
)
{
    .reg .pred %p<7>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<34>;
    .reg .f32 %f<14>;

    ld.param.u64 %rd1, [isect_vectors];
    ld.param.u64 %rd2, [isect_norms_sq];
    ld.param.u32 %r5, [isect_dims];
    ld.param.u32 %r6, [isect_n_vectors];
    ld.param.u64 %rd3, [isect_disk_queries];
    ld.param.u64 %rd4, [isect_disk_norms_sq];
    ld.param.u64 %rd5, [isect_disk_upper_sq];
    ld.param.u32 %r10, [isect_n_disks];
    ld.param.u64 %rd6, [isect_out_ids];
    ld.param.u64 %rd7, [isect_out_dists];
    ld.param.u64 %rd8, [isect_out_count];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r6;
    @%p1 ret;

    mul.wide.u32 %rd10, %r4, %r5;
    cvt.u64.u32 %rd17, %r4;
    shl.b64 %rd18, %rd17, 2;
    add.u64 %rd19, %rd2, %rd18;
    ld.global.f32 %f4, [%rd19];

    mov.u32 %r9, 0;
    mov.f32 %f10, 0f00000000;

isect_disk_loop:
    setp.ge.u32 %p2, %r9, %r10;
    @%p2 bra isect_emit;

    mul.wide.u32 %rd24, %r9, %r5;
    mov.u32 %r7, 0;
    mov.f32 %f1, 0f00000000;

isect_dim_loop:
    setp.ge.u32 %p3, %r7, %r5;
    @%p3 bra isect_disk_done;
    cvt.u64.u32 %rd11, %r7;
    add.u64 %rd12, %rd10, %rd11;
    shl.b64 %rd13, %rd12, 2;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.f32 %f2, [%rd14];
    add.u64 %rd25, %rd24, %rd11;
    shl.b64 %rd26, %rd25, 2;
    add.u64 %rd27, %rd3, %rd26;
    ld.global.f32 %f3, [%rd27];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r7, %r7, 1;
    bra isect_dim_loop;

isect_disk_done:
    cvt.u64.u32 %rd28, %r9;
    shl.b64 %rd29, %rd28, 2;
    add.u64 %rd30, %rd4, %rd29;
    ld.global.f32 %f7, [%rd30];
    add.u64 %rd31, %rd5, %rd29;
    ld.global.f32 %f9, [%rd31];
    fma.rn.f32 %f5, %f1, 0fC0000000, %f4;
    add.rn.f32 %f6, %f5, %f7;
    setp.gt.f32 %p4, %f6, %f9;
    @%p4 ret;
    setp.eq.u32 %p5, %r9, 0;
    @%p5 mov.f32 %f10, %f6;
    add.u32 %r9, %r9, 1;
    bra isect_disk_loop;

isect_emit:
    atom.global.add.u32 %r8, [%rd8], 1;
    cvt.u64.u32 %rd20, %r8;
    shl.b64 %rd21, %rd20, 2;
    add.u64 %rd22, %rd6, %rd21;
    st.global.u32 [%rd22], %r4;
    add.u64 %rd23, %rd7, %rd21;
    st.global.f32 [%rd23], %f10;
    ret;
}
"#,
    "\0"
);
