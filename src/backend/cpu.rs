use crate::backend::RingComputeBackend;
use crate::error::{Result, RingDbError};
use crate::query::Hit;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelSlice};

/// Dot product with 4 independent accumulators.
///
/// This pattern lets LLVM emit `fmla.4s` (NEON) or `vfmadd` (AVX) instead
/// of a sequential scalar `fadd` chain, yielding ~4× throughput on the
/// reduction.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        a0 = ca[0].mul_add(cb[0], a0);
        a1 = ca[1].mul_add(cb[1], a1);
        a2 = ca[2].mul_add(cb[2], a2);
        a3 = ca[3].mul_add(cb[3], a3);
    }

    let mut sum = (a0 + a1) + (a2 + a3);
    for (a, b) in rem_a.iter().zip(rem_b.iter()) {
        sum += a * b;
    }
    sum
}

/// Squared L2 norm with 4 independent accumulators.
#[inline(always)]
fn norm_sq_f32(v: &[f32]) -> f32 {
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = v.chunks_exact(4);
    let rem = chunks.remainder();

    for c in chunks {
        a0 = c[0].mul_add(c[0], a0);
        a1 = c[1].mul_add(c[1], a1);
        a2 = c[2].mul_add(c[2], a2);
        a3 = c[3].mul_add(c[3], a3);
    }

    let mut sum = (a0 + a1) + (a2 + a3);
    for x in rem {
        sum += x * x;
    }
    sum
}

/// CPU brute-force backend.
///
/// This is the reference implementation: exact float32 results that serve
/// as the ground truth for correctness testing.
pub struct CpuBackend {
    dims: usize,
    n_vectors: usize,
    vectors: Vec<f32>,  // row-major, n_vectors × dims
    norms_sq: Vec<f32>, // per-vector squared L2 norm
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            dims: 0,
            n_vectors: 0,
            vectors: Vec::new(),
            norms_sq: Vec::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RingComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()> {
        self.dims = dims;
        self.n_vectors = norms_sq.len();
        self.vectors = vectors;
        self.norms_sq = norms_sq;
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }

        let norm_sq_q = norm_sq_f32(query);
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        Ok(self
            .vectors
            .par_chunks_exact(dims)
            .zip(self.norms_sq.par_iter())
            .enumerate()
            .filter_map(|(i, (row, &norm_sq_i))| {
                let dot = dot_f32(row, query);
                let dist_sq = norm_sq_i + norm_sq_q - 2.0 * dot;
                (dist_sq >= lower_sq && dist_sq <= upper_sq).then_some(Hit {
                    id: i as u32,
                    dist_sq,
                })
            })
            .collect())
    }

    fn disk_query_f32(&self, dims: usize, query: &[f32], d_max: f32) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }

        let norm_sq_q = norm_sq_f32(query);
        let upper_sq = d_max * d_max;

        Ok(self
            .vectors
            .par_chunks_exact(dims)
            .zip(self.norms_sq.par_iter())
            .enumerate()
            .filter_map(|(i, (row, &norm_sq_i))| {
                let dot = dot_f32(row, query);
                let dist_sq = norm_sq_i + norm_sq_q - 2.0 * dot;
                (dist_sq <= upper_sq).then_some(Hit {
                    id: i as u32,
                    dist_sq,
                })
            })
            .collect())
    }

    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[(&[f32], f32)],
    ) -> Result<Vec<Hit>> {
        if self.n_vectors == 0 || disks.is_empty() {
            return Ok(Vec::new());
        }
        for &(query, _) in disks {
            if query.len() != dims {
                return Err(RingDbError::DimensionMismatch {
                    expected: dims,
                    got: query.len(),
                });
            }
        }

        let prepared: Vec<(&[f32], f32, f32)> = disks
            .iter()
            .map(|&(query, d_max)| (query, norm_sq_f32(query), d_max * d_max))
            .collect();

        Ok(self
            .vectors
            .par_chunks_exact(dims)
            .zip(self.norms_sq.par_iter())
            .enumerate()
            .filter_map(|(i, (row, &norm_sq_i))| {
                let mut first_dist_sq = 0.0;
                for (disk_i, &(query, norm_sq_q, upper_sq)) in prepared.iter().enumerate() {
                    let dot = dot_f32(row, query);
                    let dist_sq = norm_sq_i + norm_sq_q - 2.0 * dot;
                    if dist_sq > upper_sq {
                        return None;
                    }
                    if disk_i == 0 {
                        first_dist_sq = dist_sq;
                    }
                }
                Some(Hit {
                    id: i as u32,
                    dist_sq: first_dist_sq,
                })
            })
            .collect())
    }
}
