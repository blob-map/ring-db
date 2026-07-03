use crate::error::Result;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

pub(crate) struct QueryResponse {
    pub(crate) id: u32,
    pub(crate) dist_sq: f32,
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
    /// Returns all vectors with Euclidean distance to `query` in `[d_min, d_max]`,
    /// along with their squared distances.
    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Result<Vec<QueryResponse>>;

    /// Execute a float32 disk (ball) search: all vectors within radius `d_max`.
    ///
    /// This is semantically equivalent to `ring_query_f32` with `d_min = 0`.
    fn disk_query_f32(&self, dims: usize, query: &[f32], d_max: f32) -> Result<Vec<QueryResponse>>;

    /// Execute a float32 disk-intersection search.
    ///
    /// Returns all vectors that fall within every `(query, d_max)` disk. The
    /// returned distance is measured against the first disk's query vector.
    ///
    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[(&[f32], f32)],
    ) -> Result<Vec<QueryResponse>>;
}
