//! Rayon-parallel exact kernels: the correctness reference for every backend.
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::iter::{Either, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::backend::{DiskSpec, Mask, RowHit, compare_row_hits};
use crate::query::MAX_DISKS;

/// Squared L2 distance with 4 independent accumulators.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
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
fn weighted_l2_sq(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
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
    for ((a, b), weight) in rem_a.iter().zip(rem_b).zip(rem_w) {
        sum = weighted_delta_mul_add(*a, *b, *weight, sum);
    }
    sum
}

/// A zero weight skips the subtraction, so an overflowing difference in an
/// ignored dimension cannot produce `0 * inf = NaN`.
#[inline(always)]
fn weighted_delta_mul_add(a: f32, b: f32, weight: f32, sum: f32) -> f32 {
    if weight == 0.0 {
        return sum;
    }
    let delta = a - b;
    weight.mul_add(delta * delta, sum)
}

#[inline(always)]
fn distance(row: &[f32], disk: &DiskSpec<'_>) -> f32 {
    match disk.weights {
        None => l2_sq(row, disk.center),
        Some(weights) => weighted_l2_sq(row, disk.center, weights),
    }
}

#[inline(always)]
fn dense_contains(words: &[u64], row: usize) -> bool {
    words[row / 64] & (1_u64 << (row % 64)) != 0
}

/// Coordinates per Rayon job at least. Waking idle workers costs more than
/// scanning a small segment on one thread.
const MIN_JOB_VALUES: usize = 64 * 1024;

/// Eligible `(row, values)` pairs. The unfiltered path has no per-row branch;
/// `Either` dispatches once per Rayon job, not per row.
fn rows<'a>(
    vectors: &'a [f32],
    dims: usize,
    mask: Mask<'a>,
) -> impl ParallelIterator<Item = (usize, &'a [f32])> + 'a {
    let min_rows = MIN_JOB_VALUES.div_ceil(dims);
    let all = vectors
        .par_chunks_exact(dims)
        .with_min_len(min_rows)
        .enumerate();
    match mask {
        Mask::All => Either::Left(Either::Left(all)),
        Mask::Dense(words) => Either::Left(Either::Right(
            all.filter(move |(row, _)| dense_contains(words, *row)),
        )),
        Mask::Sparse(ids) => Either::Right(ids.par_iter().with_min_len(min_rows).map(move |&id| {
            let row = id as usize;
            (row, &vectors[row * dims..(row + 1) * dims])
        })),
    }
}

#[derive(Clone, Copy)]
struct Ranked(RowHit);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_row_hits(&self.0, &other.0)
    }
}

#[inline]
fn retain_top_k(heap: &mut BinaryHeap<Ranked>, hit: RowHit, k: usize) {
    let candidate = Ranked(hit);
    if heap.len() < k {
        heap.push(candidate);
    } else if heap.peek().is_some_and(|worst| candidate < *worst) {
        heap.pop();
        heap.push(candidate);
    }
}

fn collect_top_k(hits: impl ParallelIterator<Item = RowHit>, k: usize) -> Vec<RowHit> {
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
    let mut hits: Vec<RowHit> = heap.into_iter().map(|hit| hit.0).collect();
    hits.sort_unstable_by(compare_row_hits);
    hits
}

/// The `k` best rows by `(dist_sq, row)`, sorted.
pub(crate) fn knn(
    vectors: &[f32],
    dims: usize,
    query: &[f32],
    weights: Option<&[f32]>,
    k: usize,
    mask: Mask<'_>,
) -> Vec<RowHit> {
    if k == 0 || vectors.is_empty() {
        return Vec::new();
    }
    let hit = |row: usize, dist_sq: f32| RowHit {
        row: row as u32,
        dist_sq,
    };
    let rows = rows(vectors, dims, mask);
    match weights {
        None => collect_top_k(rows.map(|(row, v)| hit(row, l2_sq(v, query))), k),
        Some(w) => collect_top_k(
            rows.map(|(row, v)| hit(row, weighted_l2_sq(v, query, w))),
            k,
        ),
    }
}

/// With one disk a record is exactly a `RowHit`, collected without a copy.
fn single_disk<'a>(
    rows: impl ParallelIterator<Item = (usize, &'a [f32])>,
    base: u32,
    inside: impl Fn(&[f32]) -> Option<f32> + Sync + Send,
) -> Vec<u32> {
    let hits: Vec<RowHit> = rows
        .filter_map(|(row, v)| {
            inside(v).map(|dist_sq| RowHit {
                row: base + row as u32,
                dist_sq,
            })
        })
        .collect();
    bytemuck::allocation::cast_vec(hits)
}

/// Append one record per row inside all `disks` to `records`: the row plus
/// `base`, then one distance per disk (see `RegionResult`).
pub(crate) fn region(
    vectors: &[f32],
    dims: usize,
    base: u32,
    disks: &[DiskSpec<'_>],
    mask: Mask<'_>,
    records: &mut Vec<u32>,
) {
    if vectors.is_empty() {
        return;
    }
    let rows = rows(vectors, dims, mask);
    let found = match disks {
        [disk] if disk.weights.is_none() => single_disk(rows, base, |v| {
            let dist_sq = l2_sq(v, disk.center);
            (dist_sq <= disk.radius_sq).then_some(dist_sq)
        }),
        [disk] => single_disk(rows, base, |v| {
            let dist_sq = distance(v, disk);
            (dist_sq <= disk.radius_sq).then_some(dist_sq)
        }),
        disks => rows
            .fold(Vec::new, |mut part: Vec<u32>, (row, v)| {
                let mut dists = [0u32; MAX_DISKS];
                let inside = disks.iter().zip(&mut dists).all(|(disk, slot)| {
                    let dist_sq = distance(v, disk);
                    *slot = dist_sq.to_bits();
                    dist_sq <= disk.radius_sq
                });
                if inside {
                    part.push(base + row as u32);
                    part.extend_from_slice(&dists[..disks.len()]);
                }
                part
            })
            .reduce(Vec::new, |mut left, mut right| {
                if left.len() < right.len() {
                    std::mem::swap(&mut left, &mut right);
                }
                left.extend_from_slice(&right);
                left
            }),
    };
    if records.is_empty() {
        *records = found;
    } else {
        records.extend_from_slice(&found);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMS: usize = 2;
    // Rows at distances 0, 1, 1, 2, 3 from the origin.
    const VECTORS: [f32; 10] = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 3.0, 0.0];

    fn rows_of(hits: &[RowHit]) -> Vec<u32> {
        hits.iter().map(|hit| hit.row).collect()
    }

    /// Rows found by `region` with a base of 100, sorted.
    fn region_rows(disks: &[DiskSpec<'_>], mask: Mask<'_>) -> Vec<u32> {
        let mut records = Vec::new();
        region(&VECTORS, DIMS, 100, disks, mask, &mut records);
        let mut rows: Vec<u32> = records.chunks(1 + disks.len()).map(|r| r[0]).collect();
        rows.sort_unstable();
        rows
    }

    fn disk<'a>(center: &'a [f32], radius: f32, weights: Option<&'a [f32]>) -> DiskSpec<'a> {
        DiskSpec {
            center,
            weights,
            radius_sq: radius * radius,
        }
    }

    #[test]
    fn region_filters_every_mask_and_offsets_rows() {
        let origin = [0.0, 0.0];
        let unit = [disk(&origin, 1.0, None)];
        assert_eq!(region_rows(&unit, Mask::All), [100, 101, 102]);
        assert_eq!(region_rows(&unit, Mask::Sparse(&[1, 3, 4])), [101]);
        assert_eq!(region_rows(&unit, Mask::Dense(&[0b1_0101])), [100, 102]);
        let weighted = [disk(&origin, 2.1, Some(&[4.0, 1.0]))];
        assert_eq!(region_rows(&weighted, Mask::Sparse(&[1, 3, 4])), [101, 103]);
    }

    #[test]
    fn intersections_record_every_distance_in_disk_order() {
        let origin = [0.0, 0.0];
        let right = [1.0, 0.0];
        let weights = [1.0, 4.0];
        let disks = [disk(&origin, 2.1, Some(&weights)), disk(&right, 2.1, None)];
        assert_eq!(
            region_rows(&disks, Mask::Dense(&[0b1_1111])),
            [100, 101, 102]
        );

        let mut records = vec![7, 0, 0];
        region(&VECTORS, DIMS, 0, &disks, Mask::Sparse(&[2]), &mut records);
        assert_eq!(records, [7, 0, 0, 2, 1.0f32.to_bits(), 4.0f32.to_bits()]);
    }

    #[test]
    fn knn_is_exact_filtered_and_deterministic() {
        let origin = [0.0, 0.0];
        let hits = knn(&VECTORS, DIMS, &origin, None, 4, Mask::All);
        assert_eq!(rows_of(&hits), [0, 1, 2, 3]);
        assert_eq!(
            hits.iter().map(|hit| hit.dist_sq).collect::<Vec<_>>(),
            [0.0, 1.0, 1.0, 4.0]
        );
        let sparse = knn(&VECTORS, DIMS, &origin, None, 10, Mask::Sparse(&[2, 4]));
        assert_eq!(rows_of(&sparse), [2, 4]);
        let dense = knn(&VECTORS, DIMS, &origin, None, 1, Mask::Dense(&[0b1_1000]));
        assert_eq!(rows_of(&dense), [3]);
        let weighted = knn(&VECTORS, DIMS, &origin, Some(&[0.0, 1.0]), 3, Mask::All);
        assert_eq!(rows_of(&weighted), [0, 1, 2]);
        assert!(knn(&VECTORS, DIMS, &origin, None, 0, Mask::All).is_empty());
    }

    #[test]
    fn unweighted_knn_uses_stable_direct_deltas() {
        let vectors = [10_000_003.0, 10_000_001.0, 10_000_002.0];
        let hits = knn(&vectors, 1, &[10_000_000.0], None, 3, Mask::All);
        assert_eq!(rows_of(&hits), [1, 2, 0]);
        assert_eq!(
            hits.iter().map(|hit| hit.dist_sq).collect::<Vec<_>>(),
            [1.0, 4.0, 9.0]
        );
    }
}
