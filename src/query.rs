//! Search primitives: exact nearest neighbors and regions made of disks.
use std::sync::Arc;

use smallvec::SmallVec;

use crate::backend::DiskSpec;
use crate::error::{Result, RingDbError};
use crate::filter::Filter;
use crate::snapshot::Snapshot;

/// The largest number of disks one [`Region`] can contain.
pub const MAX_DISKS: usize = 16;

/// Stable identifier of one record.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Key(pub(crate) u64);

impl Key {
    /// Rebuild a key from the value returned by [`Key::get`].
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The integer value of this key.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Distance function of a query or disk.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Metric<'a> {
    /// `sum((x[i] - q[i])^2)`.
    #[default]
    L2,
    /// `sum(w[i] * (x[i] - q[i])^2)`. Weights are final coefficients: they are
    /// not squared, must be finite and non-negative, and zero ignores a dimension.
    WeightedL2(&'a [f32]),
}

impl<'a> Metric<'a> {
    pub(crate) const fn weights(self) -> Option<&'a [f32]> {
        match self {
            Self::L2 => None,
            Self::WeightedL2(weights) => Some(weights),
        }
    }
}

/// Exact k-nearest-neighbor search.
///
/// Returns up to `k` records ordered by ascending `(dist_sq, key)`.
#[derive(Debug, Clone, Copy)]
pub struct Knn<'a> {
    pub(crate) query: &'a [f32],
    pub(crate) k: usize,
    pub(crate) metric: Metric<'a>,
    pub(crate) filter: Option<&'a Filter>,
}

impl<'a> Knn<'a> {
    pub const fn new(query: &'a [f32], k: usize) -> Self {
        Self {
            query,
            k,
            metric: Metric::L2,
            filter: None,
        }
    }

    #[must_use]
    pub const fn metric(mut self, metric: Metric<'a>) -> Self {
        self.metric = metric;
        self
    }

    #[must_use]
    pub const fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// A ball around `center` with an inclusive `radius`.
///
/// A disk is a [`Region`] with one disk: search it directly, or combine it with
/// others through [`Disk::and`].
#[derive(Debug, Clone, Copy)]
pub struct Disk<'a> {
    pub(crate) center: &'a [f32],
    pub(crate) radius: f32,
    pub(crate) metric: Metric<'a>,
}

impl<'a> Disk<'a> {
    pub const fn new(center: &'a [f32], radius: f32) -> Self {
        Self {
            center,
            radius,
            metric: Metric::L2,
        }
    }

    #[must_use]
    pub const fn metric(mut self, metric: Metric<'a>) -> Self {
        self.metric = metric;
        self
    }

    /// Intersect this disk with `other`.
    #[must_use]
    pub fn and(self, other: Disk<'a>) -> Region<'a> {
        Region::from(self).and(other)
    }

    #[must_use]
    pub fn filter(self, filter: &'a Filter) -> Region<'a> {
        Region::from(self).filter(filter)
    }

    /// Count the records inside this disk.
    #[must_use]
    pub fn count(self) -> Count<'a> {
        Region::from(self).count()
    }
}

/// Every record inside all of its disks. It always contains at least one disk.
#[derive(Debug, Clone)]
pub struct Region<'a> {
    pub(crate) disks: SmallVec<[Disk<'a>; 1]>,
    pub(crate) filter: Option<&'a Filter>,
}

impl<'a> Region<'a> {
    /// Intersect this region with one more disk.
    #[must_use]
    pub fn and(mut self, disk: Disk<'a>) -> Self {
        self.disks.push(disk);
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn disks(&self) -> &[Disk<'a>] {
        &self.disks
    }

    /// Count the records inside this region, without returning them.
    #[must_use]
    pub fn count(self) -> Count<'a> {
        Count(self)
    }
}

/// The number of records inside a [`Region`]. Faster than the length of a
/// [`RegionResult`], because no record is returned.
///
/// ```
/// use ringdb::{Db, Disk, Options, Record, Schema};
///
/// let db = Db::in_memory(Schema::new(1), Options::default()).unwrap();
/// db.add_all([1.0, 2.0, 5.0].map(|x| Record::new([x]))).unwrap();
/// db.flush().unwrap();
/// assert_eq!(db.search(Disk::new(&[0.0], 2.0).count()).unwrap(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Count<'a>(Region<'a>);

impl<'a> From<Disk<'a>> for Region<'a> {
    fn from(disk: Disk<'a>) -> Self {
        Self {
            disks: SmallVec::from_buf([disk]),
            filter: None,
        }
    }
}

/// One nearest neighbor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub key: Key,
    /// Squared distance under the query metric.
    pub dist_sq: f32,
}

/// Resolves the rows of a search back to record keys.
///
/// Rows are numbered across the segments of a snapshot: each segment's keys
/// start at its base row.
#[derive(Debug, Clone)]
pub(crate) struct KeyLookup(Arc<[SegmentKeys]>);

#[derive(Debug)]
struct SegmentKeys {
    base: u32,
    keys: Arc<[Key]>,
    /// Set when the keys have no gaps, so that a key is computed without
    /// reading `keys`, as in a database that was only appended to.
    contiguous: bool,
}

impl KeyLookup {
    /// `segments` gives each segment's base row and its sorted keys.
    pub(crate) fn new(segments: impl IntoIterator<Item = (u32, Arc<[Key]>)>) -> Self {
        Self(
            segments
                .into_iter()
                .map(|(base, keys)| SegmentKeys {
                    base,
                    contiguous: keys
                        .last()
                        .is_some_and(|last| last.0 - keys[0].0 == keys.len() as u64 - 1),
                    keys,
                })
                .collect(),
        )
    }

    fn key(&self, row: u32) -> Key {
        let part = match &*self.0 {
            [_] => 0,
            parts => parts.partition_point(|part| part.base <= row) - 1,
        };
        let part = &self.0[part];
        let index = row - part.base;
        if part.contiguous {
            Key(part.keys[0].0 + u64::from(index))
        } else {
            part.keys[index as usize]
        }
    }
}

/// Records inside a region, in no particular order.
///
/// Each record carries one squared distance per disk, in query order. Keys are
/// resolved on access, so the result keeps the searched data alive.
#[derive(Debug, Clone)]
pub struct RegionResult {
    /// Per record: its row, then `n_disks` distance bit patterns.
    pub(crate) records: Vec<u32>,
    pub(crate) n_disks: usize,
    pub(crate) keys: KeyLookup,
}

impl RegionResult {
    pub(crate) fn new(n_disks: usize, keys: KeyLookup) -> Self {
        Self {
            records: Vec::new(),
            n_disks,
            keys,
        }
    }

    fn stride(&self) -> usize {
        1 + self.n_disks
    }

    pub fn len(&self) -> usize {
        self.records.len() / self.stride()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Number of distances per record.
    pub fn n_disks(&self) -> usize {
        self.n_disks
    }

    pub fn get(&self, index: usize) -> Option<RegionHit<'_>> {
        let start = index.checked_mul(self.stride())?;
        let record = self.records.get(start..start.checked_add(self.stride())?)?;
        Some(self.hit(record))
    }

    fn hit<'s>(&'s self, record: &'s [u32]) -> RegionHit<'s> {
        RegionHit {
            key: self.keys.key(record[0]),
            dists_sq: bytemuck::cast_slice(&record[1..]),
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RegionHit<'_>> + '_ {
        self.records
            .chunks_exact(self.stride())
            .map(|record| self.hit(record))
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = Key> + '_ {
        self.records
            .chunks_exact(self.stride())
            .map(|record| self.keys.key(record[0]))
    }
}

/// Equal when both contain the same records in the same order.
impl PartialEq for RegionResult {
    fn eq(&self, other: &Self) -> bool {
        self.n_disks == other.n_disks
            && self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

/// One record inside a region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionHit<'a> {
    pub key: Key,
    dists_sq: &'a [f32],
}

impl<'a> RegionHit<'a> {
    /// Squared distance to each disk, in query order.
    pub fn dists_sq(&self) -> &'a [f32] {
        self.dists_sq
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A search request accepted by `search`. Implemented by [`Knn`], [`Disk`],
/// [`Region`], and [`Count`].
pub trait Query: sealed::Sealed {
    type Output;

    #[doc(hidden)]
    fn execute(self, snapshot: &Snapshot) -> Result<Self::Output>;
}

impl sealed::Sealed for Knn<'_> {}
impl sealed::Sealed for Disk<'_> {}
impl sealed::Sealed for Region<'_> {}
impl sealed::Sealed for Count<'_> {}

impl Query for Knn<'_> {
    type Output = Vec<Hit>;

    fn execute(self, snapshot: &Snapshot) -> Result<Vec<Hit>> {
        snapshot.knn(&self)
    }
}

impl Query for Region<'_> {
    type Output = RegionResult;

    fn execute(self, snapshot: &Snapshot) -> Result<RegionResult> {
        snapshot.region(&self)
    }
}

impl Query for Count<'_> {
    type Output = usize;

    fn execute(self, snapshot: &Snapshot) -> Result<usize> {
        snapshot.region_count(&self.0)
    }
}

impl Query for Disk<'_> {
    type Output = RegionResult;

    fn execute(self, snapshot: &Snapshot) -> Result<RegionResult> {
        Region::from(self).execute(snapshot)
    }
}

/// Check the length and coordinates of `vector`. A coordinate that is not
/// finite is reported with `invalid`.
pub(crate) fn validate_vector(
    dims: usize,
    vector: &[f32],
    label: &str,
    invalid: fn(String) -> RingDbError,
) -> Result<()> {
    if vector.len() != dims {
        return Err(RingDbError::DimensionMismatch {
            expected: dims,
            got: vector.len(),
        });
    }
    if let Some((index, value)) = vector
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid(format!(
            "{label} coordinate at index {index} must be finite, got {value}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_metric(dims: usize, metric: Metric<'_>) -> Result<()> {
    let Some(weights) = metric.weights() else {
        return Ok(());
    };
    if weights.len() != dims {
        return Err(RingDbError::DimensionMismatch {
            expected: dims,
            got: weights.len(),
        });
    }
    if let Some((index, weight)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(RingDbError::InvalidQuery(format!(
            "weight at index {index} must be finite and non-negative, got {weight}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_disk<'a>(dims: usize, disk: &Disk<'a>) -> Result<DiskSpec<'a>> {
    validate_vector(dims, disk.center, "disk center", RingDbError::InvalidQuery)?;
    validate_metric(dims, disk.metric)?;
    // A squared radius of infinity would contain the rows whose squared
    // distance overflows, however far they are.
    let radius_sq = disk.radius * disk.radius;
    if !radius_sq.is_finite() || disk.radius < 0.0 {
        return Err(RingDbError::InvalidQuery(format!(
            "disk radius must be non-negative and its square finite, got {}",
            disk.radius
        )));
    }
    Ok(DiskSpec {
        center: disk.center,
        weights: disk.metric.weights(),
        radius_sq,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_resolve_with_and_without_gaps() {
        let keys = |values: &[u64]| -> Arc<[Key]> { values.iter().copied().map(Key).collect() };
        let lookup = KeyLookup::new([(0, keys(&[5, 6, 7])), (3, keys(&[9, 12])), (5, keys(&[20]))]);
        let found: Vec<u64> = (0..6).map(|row| lookup.key(row).0).collect();
        assert_eq!(found, [5, 6, 7, 9, 12, 20]);
        let single = KeyLookup::new([(0, keys(&[3, 8]))]);
        assert_eq!(single.key(1), Key(8));
    }
}
