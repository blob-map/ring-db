use crate::error::Result;
use crate::query::Hit;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

/// One disk passed from the public query layer to a compute backend.
#[derive(Clone, Copy)]
pub(crate) struct DiskSpec<'a> {
    pub(crate) query: &'a [f32],
    pub(crate) d_max: f32,
    pub(crate) weights: Option<&'a [f32]>,
}

/// Trait implemented by all compute backends.
///
/// The separation between upload and query reflects the core performance
/// design: the dataset is uploaded once, then many queries run against
/// the resident data without re-uploading.
pub trait RingComputeBackend: Send + Sync {
    /// Human-readable backend name (e.g. `"cpu"`).
    fn name(&self) -> &'static str;

    /// Upload a float32 dataset to the backend.
    ///
    /// - `dims`: number of dimensions per vector.
    /// - `vectors`: flat row-major buffer, length `n * dims`.
    /// - `norms_sq`: precomputed squared L2 norm per vector, length `n`.
    ///
    /// Ownership is transferred: the backend decides whether to keep or
    /// consume the data (e.g. a GPU backend would copy to VRAM and drop).
    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()>;

    /// Execute a float32 range search.
    ///
    /// Returns all vectors with query distance in `[d_min, d_max]`, along with
    /// their squared distances. `weights = None` selects ordinary L2;
    /// `Some(weights)` selects per-dimension weighted L2.
    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_min: f32,
        d_max: f32,
    ) -> Result<Vec<Hit>>;

    /// Execute a float32 disk (ball) search: all vectors within radius `d_max`.
    ///
    /// This is semantically equivalent to `ring_query_f32` with `d_min = 0`.
    fn disk_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_max: f32,
    ) -> Result<Vec<Hit>>;

    /// Execute a float32 disk-intersection search.
    ///
    /// Returns all vectors that fall within every disk. Each disk independently
    /// selects ordinary or weighted L2. The returned distance uses the first
    /// disk's metric.
    ///
    fn disk_intersection_query_f32(&self, dims: usize, disks: &[DiskSpec<'_>]) -> Result<Vec<Hit>>;
}
