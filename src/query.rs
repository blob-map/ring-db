//! Search primitives: exact nearest neighbors and regions made of disks.
use std::sync::Arc;

use smallvec::SmallVec;

use crate::error::Result;
use crate::metadata::CandidateSet;

/// The largest number of disks one [`Region`] can contain.
pub const MAX_DISKS: usize = 16;

/// Stable identifier of one record.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Key(pub(crate) u64);

impl Key {
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
    pub(crate) filter: Option<&'a CandidateSet>,
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
    pub const fn filter(mut self, filter: &'a CandidateSet) -> Self {
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
    pub fn filter(self, filter: &'a CandidateSet) -> Region<'a> {
        Region::from(self).filter(filter)
    }
}

/// Every record inside all of its disks. It always contains at least one disk.
#[derive(Debug, Clone)]
pub struct Region<'a> {
    pub(crate) disks: SmallVec<[Disk<'a>; 1]>,
    pub(crate) filter: Option<&'a CandidateSet>,
}

impl<'a> Region<'a> {
    /// Intersect this region with one more disk.
    #[must_use]
    pub fn and(mut self, disk: Disk<'a>) -> Self {
        self.disks.push(disk);
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: &'a CandidateSet) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn disks(&self) -> &[Disk<'a>] {
        &self.disks
    }
}

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
#[derive(Debug, Clone)]
pub(crate) struct KeyLookup(pub(crate) Arc<[Key]>);

impl KeyLookup {
    fn key(&self, row: u32) -> Key {
        self.0[row as usize]
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
        let record = self.records.get(start..start + self.stride())?;
        Some(RegionHit {
            key: self.keys.key(record[0]),
            dists_sq: bytemuck::cast_slice(&record[1..]),
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RegionHit<'_>> + '_ {
        (0..self.len()).map(|index| self.get(index).expect("index is in bounds"))
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = Key> + '_ {
        self.iter().map(|hit| hit.key)
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
/// and [`Region`].
pub trait Query: sealed::Sealed {
    type Output;

    #[doc(hidden)]
    fn execute(self, searcher: &crate::engine::Searcher<'_>) -> Result<Self::Output>;
}

impl sealed::Sealed for Knn<'_> {}
impl sealed::Sealed for Disk<'_> {}
impl sealed::Sealed for Region<'_> {}

impl Query for Knn<'_> {
    type Output = Vec<Hit>;

    fn execute(self, searcher: &crate::engine::Searcher<'_>) -> Result<Vec<Hit>> {
        searcher.knn(&self)
    }
}

impl Query for Region<'_> {
    type Output = RegionResult;

    fn execute(self, searcher: &crate::engine::Searcher<'_>) -> Result<RegionResult> {
        searcher.region(&self)
    }
}

impl Query for Disk<'_> {
    type Output = RegionResult;

    fn execute(self, searcher: &crate::engine::Searcher<'_>) -> Result<RegionResult> {
        Region::from(self).execute(searcher)
    }
}
