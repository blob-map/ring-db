use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelSlice};

use crate::backend::{DiskSpec, RingComputeBackend};
use crate::error::{Result, RingDbError};
use crate::metadata::CandidateSetView;
use crate::query::Hit;

/// Squared L2 distance with 4 independent accumulators.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        a0 = d0.mul_add(d0, a0);
        a1 = d1.mul_add(d1, a1);
        a2 = d2.mul_add(d2, a2);
        a3 = d3.mul_add(d3, a3);
    }

    let mut sum = (a0 + a1) + (a2 + a3);
    for (a, b) in rem_a.iter().zip(rem_b) {
        let delta = a - b;
        sum = delta.mul_add(delta, sum);
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
        a0 = weighted_delta_mul_add(ca[0], cb[0], cw[0], a0);
        a1 = weighted_delta_mul_add(ca[1], cb[1], cw[1], a1);
        a2 = weighted_delta_mul_add(ca[2], cb[2], cw[2], a2);
        a3 = weighted_delta_mul_add(ca[3], cb[3], cw[3], a3);
    }

    let mut sum = (a0 + a1) + (a2 + a3);
    for ((a, b), weight) in rem_a.iter().zip(rem_b.iter()).zip(rem_w.iter()) {
        sum = weighted_delta_mul_add(*a, *b, *weight, sum);
    }
    sum
}

#[inline(always)]
fn weighted_delta_mul_add(a: f32, b: f32, weight: f32, sum: f32) -> f32 {
    if weight == 0.0 {
        return sum;
    }
    let delta = a - b;
    weight.mul_add(delta * delta, sum)
}

#[inline(always)]
fn dense_contains(words: &[u64], id: usize) -> bool {
    words[id / 64] & (1_u64 << (id % 64)) != 0
}

fn compare_hits(left: &Hit, right: &Hit) -> Ordering {
    left.dist_sq
        .total_cmp(&right.dist_sq)
        .then_with(|| left.id.cmp(&right.id))
}

#[derive(Clone, Copy)]
struct RankedHit(Hit);

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_hits(&self.0, &other.0)
    }
}

#[inline]
fn retain_top_k(heap: &mut BinaryHeap<RankedHit>, hit: Hit, k: usize) {
    let candidate = RankedHit(hit);
    if heap.len() < k {
        heap.push(candidate);
        return;
    }

    if heap.peek().is_some_and(|worst| candidate < *worst) {
        heap.pop();
        heap.push(candidate);
    }
}

fn collect_top_k<I>(hits: I, k: usize) -> Vec<Hit>
where
    I: ParallelIterator<Item = Hit>,
{
    debug_assert!(k > 0);
    let heap = hits
        .fold(BinaryHeap::new, |mut heap, hit| {
            retain_top_k(&mut heap, hit, k);
            heap
        })
        .reduce(BinaryHeap::new, |mut left, right| {
            for hit in right {
                retain_top_k(&mut left, hit.0, k);
            }
            left
        });

    let mut hits: Vec<Hit> = heap.into_iter().map(|hit| hit.0).collect();
    hits.sort_unstable_by(compare_hits);
    hits
}

struct PreparedUnweightedDisk<'a> {
    query: &'a [f32],
    upper_sq: f32,
}

enum PreparedDisk<'a> {
    Unweighted {
        query: &'a [f32],
        upper_sq: f32,
    },
    Weighted {
        query: &'a [f32],
        weights: &'a [f32],
        upper_sq: f32,
    },
}

#[inline(always)]
fn unweighted_intersection_hit(
    id: usize,
    row: &[f32],
    disks: &[PreparedUnweightedDisk<'_>],
) -> Option<Hit> {
    let mut first_dist_sq = 0.0;
    for (disk_i, disk) in disks.iter().enumerate() {
        let dist_sq = l2_sq_f32(row, disk.query);
        if dist_sq > disk.upper_sq {
            return None;
        }
        if disk_i == 0 {
            first_dist_sq = dist_sq;
        }
    }
    Some(Hit {
        id: id as u32,
        dist_sq: first_dist_sq,
    })
}

#[inline(always)]
fn mixed_intersection_hit(id: usize, row: &[f32], disks: &[PreparedDisk<'_>]) -> Option<Hit> {
    let mut first_dist_sq = 0.0;
    for (disk_i, disk) in disks.iter().enumerate() {
        let (dist_sq, upper_sq) = match disk {
            PreparedDisk::Unweighted { query, upper_sq } => (l2_sq_f32(row, query), *upper_sq),
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
        id: id as u32,
        dist_sq: first_dist_sq,
    })
}

/// CPU brute-force backend.
///
/// This is the reference implementation: exact float32 results that serve
/// as the ground truth for correctness testing.
pub struct CpuBackend {
    n_vectors: usize,
    vectors: Vec<f32>, // row-major, n_vectors × dims
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            n_vectors: 0,
            vectors: Vec::new(),
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

    fn filtered_query<F>(
        &self,
        dims: usize,
        candidates: CandidateSetView<'_>,
        evaluate: F,
    ) -> Vec<Hit>
    where
        F: Fn(usize, &[f32]) -> Option<Hit> + Sync,
    {
        match candidates {
            CandidateSetView::All => self
                .vectors
                .par_chunks_exact(dims)
                .enumerate()
                .filter_map(|(id, row)| evaluate(id, row))
                .collect(),
            CandidateSetView::Empty => Vec::new(),
            CandidateSetView::Sparse(ids) => ids
                .par_iter()
                .filter_map(|&id| {
                    let id = id as usize;
                    let start = id * dims;
                    evaluate(id, &self.vectors[start..start + dims])
                })
                .collect(),
            CandidateSetView::Dense(words) => self
                .vectors
                .par_chunks_exact(dims)
                .enumerate()
                .filter_map(|(id, row)| {
                    if dense_contains(words, id) {
                        evaluate(id, row)
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    fn candidate_top_k<F>(
        &self,
        dims: usize,
        k: usize,
        candidates: CandidateSetView<'_>,
        distance: F,
    ) -> Vec<Hit>
    where
        F: Fn(&[f32]) -> f32 + Sync,
    {
        match candidates {
            CandidateSetView::All => collect_top_k(
                self.vectors
                    .par_chunks_exact(dims)
                    .enumerate()
                    .map(|(id, row)| Hit {
                        id: id as u32,
                        dist_sq: distance(row),
                    }),
                k,
            ),
            CandidateSetView::Empty => Vec::new(),
            CandidateSetView::Sparse(ids) => collect_top_k(
                ids.par_iter().map(|&id| {
                    let index = id as usize;
                    let start = index * dims;
                    Hit {
                        id,
                        dist_sq: distance(&self.vectors[start..start + dims]),
                    }
                }),
                k,
            ),
            CandidateSetView::Dense(words) => collect_top_k(
                self.vectors
                    .par_chunks_exact(dims)
                    .enumerate()
                    .filter_map(|(id, row)| {
                        dense_contains(words, id).then(|| Hit {
                            id: id as u32,
                            dist_sq: distance(row),
                        })
                    }),
                k,
            ),
        }
    }

    #[inline(always)]
    fn ring_query_unweighted(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
    ) -> Vec<Hit> {
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(i, row)| {
                let dist_sq = l2_sq_f32(row, query);
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

    fn ring_query_filtered_unweighted(
        &self,
        dims: usize,
        query: &[f32],
        d_min: f32,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        self.filtered_query(dims, candidates, |id, row| {
            let dist_sq = l2_sq_f32(row, query);
            (dist_sq >= lower_sq && dist_sq <= upper_sq).then_some(Hit {
                id: id as u32,
                dist_sq,
            })
        })
    }

    fn ring_query_filtered_weighted(
        &self,
        dims: usize,
        query: &[f32],
        weights: &[f32],
        d_min: f32,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        let lower_sq = d_min * d_min;
        let upper_sq = d_max * d_max;

        self.filtered_query(dims, candidates, |id, row| {
            let dist_sq = weighted_l2_sq_f32(row, query, weights);
            (dist_sq >= lower_sq && dist_sq <= upper_sq).then_some(Hit {
                id: id as u32,
                dist_sq,
            })
        })
    }

    #[inline(always)]
    fn disk_query_unweighted(&self, dims: usize, query: &[f32], d_max: f32) -> Vec<Hit> {
        let upper_sq = d_max * d_max;

        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(i, row)| {
                let dist_sq = l2_sq_f32(row, query);
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

    fn disk_query_filtered_unweighted(
        &self,
        dims: usize,
        query: &[f32],
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        let upper_sq = d_max * d_max;

        self.filtered_query(dims, candidates, |id, row| {
            let dist_sq = l2_sq_f32(row, query);
            (dist_sq <= upper_sq).then_some(Hit {
                id: id as u32,
                dist_sq,
            })
        })
    }

    fn disk_query_filtered_weighted(
        &self,
        dims: usize,
        query: &[f32],
        weights: &[f32],
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        let upper_sq = d_max * d_max;

        self.filtered_query(dims, candidates, |id, row| {
            let dist_sq = weighted_l2_sq_f32(row, query, weights);
            (dist_sq <= upper_sq).then_some(Hit {
                id: id as u32,
                dist_sq,
            })
        })
    }

    #[inline(always)]
    fn disk_intersection_unweighted(
        &self,
        dims: usize,
        disks: &[PreparedUnweightedDisk<'_>],
    ) -> Vec<Hit> {
        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(id, row)| unweighted_intersection_hit(id, row, disks))
            .collect()
    }

    fn disk_intersection_mixed(&self, dims: usize, disks: &[PreparedDisk<'_>]) -> Vec<Hit> {
        self.vectors
            .par_chunks_exact(dims)
            .enumerate()
            .filter_map(|(id, row)| mixed_intersection_hit(id, row, disks))
            .collect()
    }

    fn disk_intersection_filtered_unweighted(
        &self,
        dims: usize,
        disks: &[PreparedUnweightedDisk<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        self.filtered_query(dims, candidates, |id, row| {
            unweighted_intersection_hit(id, row, disks)
        })
    }

    fn disk_intersection_filtered_mixed(
        &self,
        dims: usize,
        disks: &[PreparedDisk<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        self.filtered_query(dims, candidates, |id, row| {
            mixed_intersection_hit(id, row, disks)
        })
    }

    fn knn_query_unweighted(
        &self,
        dims: usize,
        query: &[f32],
        k: usize,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        self.candidate_top_k(dims, k, candidates, |row| l2_sq_f32(row, query))
    }

    fn knn_query_weighted(
        &self,
        dims: usize,
        query: &[f32],
        weights: &[f32],
        k: usize,
        candidates: CandidateSetView<'_>,
    ) -> Vec<Hit> {
        self.candidate_top_k(dims, k, candidates, |row| {
            weighted_l2_sq_f32(row, query, weights)
        })
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

    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>) -> Result<()> {
        self.n_vectors = vectors.len() / dims;
        self.vectors = vectors;
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
        Self::validate_query_shape(dims, query, weights)?;
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }

        Ok(match (weights, candidates) {
            (_, CandidateSetView::Empty) => Vec::new(),
            (None, CandidateSetView::All) => self.ring_query_unweighted(dims, query, d_min, d_max),
            (Some(weights), CandidateSetView::All) => {
                self.ring_query_weighted(dims, query, weights, d_min, d_max)
            }
            (None, candidates) => {
                self.ring_query_filtered_unweighted(dims, query, d_min, d_max, candidates)
            }
            (Some(weights), candidates) => {
                self.ring_query_filtered_weighted(dims, query, weights, d_min, d_max, candidates)
            }
        })
    }

    fn disk_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        d_max: f32,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        Self::validate_query_shape(dims, query, weights)?;
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }

        Ok(match (weights, candidates) {
            (_, CandidateSetView::Empty) => Vec::new(),
            (None, CandidateSetView::All) => self.disk_query_unweighted(dims, query, d_max),
            (Some(weights), CandidateSetView::All) => {
                self.disk_query_weighted(dims, query, weights, d_max)
            }
            (None, candidates) => {
                self.disk_query_filtered_unweighted(dims, query, d_max, candidates)
            }
            (Some(weights), candidates) => {
                self.disk_query_filtered_weighted(dims, query, weights, d_max, candidates)
            }
        })
    }

    fn disk_intersection_query_f32(
        &self,
        dims: usize,
        disks: &[DiskSpec<'_>],
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        Self::validate_disk_shapes(dims, disks)?;
        if self.n_vectors == 0 || disks.is_empty() || matches!(candidates, CandidateSetView::Empty)
        {
            return Ok(Vec::new());
        }

        if disks.iter().all(|disk| disk.weights.is_none()) {
            let prepared: Vec<PreparedUnweightedDisk<'_>> = disks
                .iter()
                .map(|disk| PreparedUnweightedDisk {
                    query: disk.query,
                    upper_sq: disk.d_max * disk.d_max,
                })
                .collect();
            return Ok(match candidates {
                CandidateSetView::All => self.disk_intersection_unweighted(dims, &prepared),
                candidates => {
                    self.disk_intersection_filtered_unweighted(dims, &prepared, candidates)
                }
            });
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
                    upper_sq: disk.d_max * disk.d_max,
                },
            })
            .collect();
        Ok(match candidates {
            CandidateSetView::All => self.disk_intersection_mixed(dims, &prepared),
            candidates => self.disk_intersection_filtered_mixed(dims, &prepared, candidates),
        })
    }

    fn knn_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        k: usize,
        candidates: CandidateSetView<'_>,
    ) -> Result<Vec<Hit>> {
        Self::validate_query_shape(dims, query, weights)?;
        if self.n_vectors == 0 || k == 0 || matches!(candidates, CandidateSetView::Empty) {
            return Ok(Vec::new());
        }

        Ok(match weights {
            None => self.knn_query_unweighted(dims, query, k, candidates),
            Some(weights) => self.knn_query_weighted(dims, query, weights, k, candidates),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMS: usize = 2;

    fn backend_from_vectors(dims: usize, vectors: Vec<f32>) -> CpuBackend {
        let mut backend = CpuBackend::new();
        backend.upload_f32_dataset(dims, vectors).unwrap();
        backend
    }

    fn test_backend() -> CpuBackend {
        backend_from_vectors(
            DIMS,
            vec![0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
        )
    }

    fn ids(hits: &[Hit]) -> Vec<u32> {
        hits.iter().map(|hit| hit.id).collect()
    }

    fn scalar_knn(
        vectors: &[f32],
        dims: usize,
        query: &[f32],
        weights: Option<&[f32]>,
        candidates: &[u32],
        k: usize,
    ) -> Vec<Hit> {
        let mut hits: Vec<_> = candidates
            .iter()
            .map(|&id| {
                let start = id as usize * dims;
                let row = &vectors[start..start + dims];
                let dist_sq = row
                    .iter()
                    .zip(query)
                    .enumerate()
                    .map(|(dimension, (&value, &target))| {
                        let delta = value - target;
                        weights.map_or(delta * delta, |weights| weights[dimension] * delta * delta)
                    })
                    .sum();
                Hit { id, dist_sq }
            })
            .collect();
        hits.sort_unstable_by(compare_hits);
        hits.truncate(k);
        hits
    }

    #[test]
    fn candidate_representations_filter_every_range_query() {
        let backend = test_backend();
        let origin = [0.0, 0.0];

        let ring = backend
            .ring_query_f32(
                DIMS,
                &origin,
                None,
                0.5,
                2.1,
                CandidateSetView::Sparse(&[0, 1, 3, 4]),
            )
            .unwrap();
        assert_eq!(ids(&ring), [1, 3]);

        let weighted_ring = backend
            .ring_query_f32(
                DIMS,
                &origin,
                Some(&[4.0, 1.0]),
                1.5,
                2.1,
                CandidateSetView::Dense(&[0b1_1111]),
            )
            .unwrap();
        assert_eq!(ids(&weighted_ring), [1, 2, 3]);

        let disk = backend
            .disk_query_f32(
                DIMS,
                &origin,
                None,
                1.1,
                CandidateSetView::Dense(&[0b1_0101]),
            )
            .unwrap();
        assert_eq!(ids(&disk), [0, 2]);

        let weighted_disk = backend
            .disk_query_f32(
                DIMS,
                &origin,
                Some(&[4.0, 1.0]),
                2.1,
                CandidateSetView::Sparse(&[1, 3, 4]),
            )
            .unwrap();
        assert_eq!(ids(&weighted_disk), [1, 3]);
    }

    #[test]
    fn candidates_filter_unweighted_and_mixed_disk_intersections() {
        let backend = test_backend();
        let origin = [0.0, 0.0];
        let right = [1.0, 0.0];
        let unweighted = [
            DiskSpec {
                query: &origin,
                d_max: 2.1,
                weights: None,
            },
            DiskSpec {
                query: &right,
                d_max: 2.1,
                weights: None,
            },
        ];
        let hits = backend
            .disk_intersection_query_f32(DIMS, &unweighted, CandidateSetView::Sparse(&[0, 2, 3, 4]))
            .unwrap();
        assert_eq!(ids(&hits), [0, 2]);

        let weights = [1.0, 4.0];
        let mixed = [
            DiskSpec {
                query: &origin,
                d_max: 2.1,
                weights: Some(&weights),
            },
            DiskSpec {
                query: &right,
                d_max: 2.1,
                weights: None,
            },
        ];
        let hits = backend
            .disk_intersection_query_f32(DIMS, &mixed, CandidateSetView::Dense(&[0b1_1111]))
            .unwrap();
        assert_eq!(ids(&hits), [0, 1, 2]);
    }

    #[test]
    fn knn_is_exact_filtered_and_deterministic() {
        let backend = test_backend();
        let origin = [0.0, 0.0];

        let hits = backend
            .knn_query_f32(DIMS, &origin, None, 4, CandidateSetView::All)
            .unwrap();
        assert_eq!(ids(&hits), [0, 1, 2, 3]);
        assert_eq!(
            hits.iter().map(|hit| hit.dist_sq).collect::<Vec<_>>(),
            [0.0, 1.0, 1.0, 4.0]
        );

        let sparse = backend
            .knn_query_f32(DIMS, &origin, None, 10, CandidateSetView::Sparse(&[2, 4]))
            .unwrap();
        assert_eq!(ids(&sparse), [2, 4]);

        let dense = backend
            .knn_query_f32(DIMS, &origin, None, 1, CandidateSetView::Dense(&[0b1_1000]))
            .unwrap();
        assert_eq!(ids(&dense), [3]);

        let weighted = backend
            .knn_query_f32(DIMS, &origin, Some(&[0.0, 1.0]), 3, CandidateSetView::All)
            .unwrap();
        assert_eq!(ids(&weighted), [0, 1, 2]);

        assert!(
            backend
                .knn_query_f32(DIMS, &origin, None, 0, CandidateSetView::All)
                .unwrap()
                .is_empty()
        );
        assert!(
            backend
                .knn_query_f32(DIMS, &origin, None, 1, CandidateSetView::Empty)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unweighted_knn_uses_stable_direct_deltas() {
        let backend = backend_from_vectors(1, vec![10_000_003.0, 10_000_001.0, 10_000_002.0]);
        let hits = backend
            .knn_query_f32(1, &[10_000_000.0], None, 3, CandidateSetView::All)
            .unwrap();

        assert_eq!(ids(&hits), [1, 2, 0]);
        assert_eq!(
            hits.iter().map(|hit| hit.dist_sq).collect::<Vec<_>>(),
            [1.0, 4.0, 9.0]
        );
    }

    #[test]
    fn randomized_knn_matches_scalar_reference_for_every_candidate_path() {
        const RANDOM_DIMS: usize = 7;
        const N_VECTORS: usize = 257;

        let query: Vec<_> = (0..RANDOM_DIMS)
            .map(|dimension| 1_000_000.0 + dimension as f32 * 1_000.0)
            .collect();
        let mut vectors = Vec::with_capacity(N_VECTORS * RANDOM_DIMS);
        let mut positive_tie = query.clone();
        positive_tie[0] += 1.0;
        vectors.extend_from_slice(&positive_tie);
        let mut negative_tie = query.clone();
        negative_tie[0] -= 1.0;
        vectors.extend_from_slice(&negative_tie);

        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for _ in 2..N_VECTORS {
            for &target in &query {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let magnitude = 2.0 + ((state >> 32) as u32 % 8_001) as f32 / 1_000.0;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                vectors.push(target + sign * magnitude);
            }
        }

        let backend = backend_from_vectors(RANDOM_DIMS, vectors.clone());
        let all_ids: Vec<u32> = (0..N_VECTORS as u32).collect();
        let sparse_ids = vec![0, 1, 5, 64, 129, 200, 256];
        let mut dense_words = vec![0_u64; N_VECTORS.div_ceil(64)];
        let dense_ids: Vec<u32> = (0..N_VECTORS as u32)
            .filter(|&id| id % 2 == 0 || id == 1)
            .inspect(|&id| dense_words[id as usize / 64] |= 1_u64 << (id % 64))
            .collect();
        let candidate_cases = [
            ("all", CandidateSetView::All, all_ids.as_slice()),
            (
                "sparse",
                CandidateSetView::Sparse(&sparse_ids),
                sparse_ids.as_slice(),
            ),
            (
                "dense",
                CandidateSetView::Dense(&dense_words),
                dense_ids.as_slice(),
            ),
        ];
        let weights = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

        for (candidate_name, candidate_view, candidate_ids) in candidate_cases {
            for query_weights in [None, Some(weights.as_slice())] {
                for k in [0, 1, 4, candidate_ids.len() + 3] {
                    let expected = scalar_knn(
                        &vectors,
                        RANDOM_DIMS,
                        &query,
                        query_weights,
                        candidate_ids,
                        k,
                    );
                    let actual = backend
                        .knn_query_f32(RANDOM_DIMS, &query, query_weights, k, candidate_view)
                        .unwrap();

                    assert_eq!(
                        ids(&actual),
                        ids(&expected),
                        "candidate path={candidate_name}, weighted={}, k={k}",
                        query_weights.is_some()
                    );
                    for (actual, expected) in actual.iter().zip(&expected) {
                        let tolerance = 1.0e-5 * expected.dist_sq.abs().max(1.0);
                        assert!(
                            (actual.dist_sq - expected.dist_sq).abs() <= tolerance,
                            "candidate path={candidate_name}, weighted={}, k={k}, id={}, actual={}, expected={}",
                            query_weights.is_some(),
                            actual.id,
                            actual.dist_sq,
                            expected.dist_sq,
                        );
                    }
                }
            }
        }
    }
}
