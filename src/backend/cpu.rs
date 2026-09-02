use crate::backend::{DiskSpec, RingComputeBackend};
use crate::error::{Result, RingDbError};
use crate::query::Hit;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelSlice};

// `slice::as_chunks` requires a newer compiler than Rust 2024's baseline.
// Keep the equivalent iterator form without raising ring-db's implicit MSRV.
#[allow(clippy::chunks_exact_to_as_chunks)]
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
#[allow(clippy::chunks_exact_to_as_chunks)]
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

/// Squared weighted L2 distance with 4 independent accumulators.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn weighted_l2_sq_f32(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), weights.len());

    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let chunks_w = weights.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    let rem_w = chunks_w.remainder();

    for ((ca, cb), cw) in chunks_a.zip(chunks_b).zip(chunks_w) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        a0 = cw[0].mul_add(d0 * d0, a0);
        a1 = cw[1].mul_add(d1 * d1, a1);
        a2 = cw[2].mul_add(d2 * d2, a2);
        a3 = cw[3].mul_add(d3 * d3, a3);
    }

    let mut sum = (a0 + a1) + (a2 + a3);
    for ((a, b), weight) in rem_a.iter().zip(rem_b.iter()).zip(rem_w.iter()) {
        let delta = a - b;
        sum = weight.mul_add(delta * delta, sum);
    }
    sum
}

struct PreparedUnweightedDisk<'a> {
    query: &'a [f32],
    norm_sq_q: f32,
    upper_sq: f32,
}

enum PreparedDisk<'a> {
    Unweighted {
        query: &'a [f32],
        norm_sq_q: f32,
        upper_sq: f32,
    },
    Weighted {
        query: &'a [f32],
        weights: &'a [f32],
        upper_sq: f32,
    },
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

    fn validate_query_shape(dims: usize, query: &[f32], weights: Option<&[f32]>) -> Result<()> {
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }
        if let Some(weights) = weights.filter(|weights| weights.len() != dims) {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: weights.len(),
            });
        }
        Ok(())
    }

    fn validate_disk_shapes(dims: usize, disks: &[DiskSpec<'_>]) -> Result<()> {
        for disk in disks {
            Self::validate_query_shape(dims, disk.query, disk.weights)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn ring_query_unweighted(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Vec<Hit> {
        let norm_sq_q = norm_sq_f32(query);
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        self.vectors
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
            .collect()
    }

    fn ring_query_weighted(
        &self,
        dims: usize,
        query: &[f32],
        weights: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Vec<Hit> {
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(i, row)| {
                let dist_sq = weighted_l2_sq_f32(row, query, weights);
                (dist_sq >= lower_sq && dist_sq <= upper_sq).then_some(Hit {
                    id: i as u32,
                    dist_sq,
                })
            })
            .collect()
    }

    #[inline(always)]
    fn disk_query_unweighted(&self, dims: usize, query: &[f32], d_max: f32) -> Vec<Hit> {
        let norm_sq_q = norm_sq_f32(query);
        let upper_sq = d_max * d_max;

        self.vectors
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
            .collect()
    }

    fn disk_query_weighted(
        &self,
        dims: usize,
        query: &[f32],
        weights: &[f32],
        d_max: f32,
    ) -> Vec<Hit> {
        let upper_sq = d_max * d_max;

        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(i, row)| {
                let dist_sq = weighted_l2_sq_f32(row, query, weights);
                (dist_sq <= upper_sq).then_some(Hit {
                    id: i as u32,
                    dist_sq,
                })
            })
            .collect()
    }

    #[inline(always)]
    fn disk_intersection_unweighted(
        &self,
        dims: usize,
        disks: &[PreparedUnweightedDisk<'_>],
    ) -> Vec<Hit> {
        self.vectors
            .par_chunks_exact(dims)
            .zip(self.norms_sq.par_iter())
            .enumerate()
            .filter_map(|(i, (row, &norm_sq_i))| {
                let mut first_dist_sq = 0.0;
                for (disk_i, disk) in disks.iter().enumerate() {
                    let dot = dot_f32(row, disk.query);
                    let dist_sq = norm_sq_i + disk.norm_sq_q - 2.0 * dot;
                    if dist_sq > disk.upper_sq {
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
            .collect()
    }

    fn disk_intersection_mixed(&self, dims: usize, disks: &[PreparedDisk<'_>]) -> Vec<Hit> {
        self.vectors
            .par_chunks_exact(dims)
            .zip(self.norms_sq.par_iter())
            .enumerate()
            .filter_map(|(i, (row, &norm_sq_i))| {
                let mut first_dist_sq = 0.0;
                for (disk_i, disk) in disks.iter().enumerate() {
                    let (dist_sq, upper_sq) = match disk {
                        PreparedDisk::Unweighted {
                            query,
                            norm_sq_q,
                            upper_sq,
                        } => (norm_sq_i + norm_sq_q - 2.0 * dot_f32(row, query), *upper_sq),
                        PreparedDisk::Weighted {
                            query,
                            weights,
                            upper_sq,
                        } => (weighted_l2_sq_f32(row, query, weights), *upper_sq),
                    };
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
            .collect()
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
        weights: Option<&[f32]>,
        d_min: f32,
        d_max: f32,
    ) -> Result<Vec<Hit>> {
        Self::validate_query_shape(dims, query, weights)?;
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }

        Ok(match weights {
            Some(weights) => self.ring_query_weighted(dims, query, weights, d_min, d_max),
            None => self.ring_query_unweighted(dims, query, d_min, d_max),
        })
    }

    fn disk_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_max: f32,
    ) -> Result<Vec<Hit>> {
        Self::validate_query_shape(dims, query, weights)?;
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }

        Ok(match weights {
            Some(weights) => self.disk_query_weighted(dims, query, weights, d_max),
            None => self.disk_query_unweighted(dims, query, d_max),
        })
    }

    fn disk_intersection_query_f32(&self, dims: usize, disks: &[DiskSpec<'_>]) -> Result<Vec<Hit>> {
        Self::validate_disk_shapes(dims, disks)?;
        if self.n_vectors == 0 || disks.is_empty() {
            return Ok(Vec::new());
        }

        if disks.iter().all(|disk| disk.weights.is_none()) {
            let prepared: Vec<PreparedUnweightedDisk<'_>> = disks
                .iter()
                .map(|disk| PreparedUnweightedDisk {
                    query: disk.query,
                    norm_sq_q: norm_sq_f32(disk.query),
                    upper_sq: disk.d_max * disk.d_max,
                })
                .collect();
            return Ok(self.disk_intersection_unweighted(dims, &prepared));
        }

        let prepared: Vec<PreparedDisk<'_>> = disks
            .iter()
            .map(|disk| match disk.weights {
                Some(weights) => PreparedDisk::Weighted {
                    query: disk.query,
                    weights,
                    upper_sq: disk.d_max * disk.d_max,
                },
                None => PreparedDisk::Unweighted {
                    query: disk.query,
                    norm_sq_q: norm_sq_f32(disk.query),
                    upper_sq: disk.d_max * disk.d_max,
                },
            })
            .collect();
        Ok(self.disk_intersection_mixed(dims, &prepared))
    }
}
