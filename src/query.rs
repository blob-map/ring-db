use std::time::Duration;

use crate::metadata::CandidateSet;

/// A single match returned by a query.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Insertion-order ID of the matching vector (first inserted = 0).
    pub id: u32,
    /// Squared distance under the metric used by the query.
    ///
    /// This is squared L2 distance for an unweighted query and squared weighted
    /// L2 distance for a weighted query. For a [`DiskIntersectionQuery`], it is
    /// measured using the first disk's metric.
    pub dist_sq: f32,
}

/// A ring query: find all vectors whose query distance to `query`
/// lies within `[d - lambda, d + lambda]`.
///
/// Internally, ringdb compares squared query distances to avoid computing
/// square roots. The ring bounds become:
///
/// ```text
/// lower_sq = max(0, d - lambda)²
/// upper_sq = (d + lambda)²
/// ```
pub struct RingQuery<'a> {
    /// The query vector. Must have the database dimension and finite values.
    pub(crate) query: &'a [f32],
    /// Finite, non-negative target distance (centre of the ring).
    pub(crate) d: f32,
    /// Finite, non-negative half-width of the ring.
    pub(crate) lambda: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
    /// Optional set of vector IDs eligible to match.
    pub(crate) candidates: Option<&'a CandidateSet>,
}

impl<'a> RingQuery<'a> {
    /// Create an unweighted ring query.
    pub const fn new(query: &'a [f32], d: f32, lambda: f32) -> Self {
        Self {
            query,
            d,
            lambda,
            weights: None,
            candidates: None,
        }
    }

    /// Use squared weighted L2 distance for this query.
    ///
    /// Each value is the final coefficient in
    /// `sum(weights[i] * (x[i] - query[i])^2)`. Weights are not squared by
    /// ring-db. They are validated when the query is executed.
    #[must_use]
    pub const fn with_weights(mut self, weights: &'a [f32]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Restrict this query to the vectors in `candidates`.
    #[must_use]
    pub const fn with_candidates(mut self, candidates: &'a CandidateSet) -> Self {
        self.candidates = Some(candidates);
        self
    }

    /// Query vector.
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    /// Target distance (centre of the ring).
    pub const fn d(&self) -> f32 {
        self.d
    }

    /// Half-width of the ring.
    pub const fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Optional per-dimension weighted-L2 coefficients.
    pub const fn weights(&self) -> Option<&'a [f32]> {
        self.weights
    }

    /// Optional set of vector IDs eligible to match.
    pub const fn candidates(&self) -> Option<&'a CandidateSet> {
        self.candidates
    }
}

/// A range query: find all vectors whose query distance to `query`
/// lies within `[d_min, d_max]`.
pub struct RangeQuery<'a> {
    /// The query vector. Must have the database dimension and finite values.
    pub(crate) query: &'a [f32],
    /// Inclusive lower distance bound. Must be finite and non-negative.
    pub(crate) d_min: f32,
    /// Inclusive upper distance bound. Must be finite and at least `d_min`.
    pub(crate) d_max: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
    /// Optional set of vector IDs eligible to match.
    pub(crate) candidates: Option<&'a CandidateSet>,
}

impl<'a> RangeQuery<'a> {
    /// Create an unweighted range query.
    pub const fn new(query: &'a [f32], d_min: f32, d_max: f32) -> Self {
        Self {
            query,
            d_min,
            d_max,
            weights: None,
            candidates: None,
        }
    }

    /// Use squared weighted L2 distance for this query.
    ///
    /// Each value is the final coefficient in
    /// `sum(weights[i] * (x[i] - query[i])^2)`. Weights are not squared by
    /// ring-db. They are validated when the query is executed.
    #[must_use]
    pub const fn with_weights(mut self, weights: &'a [f32]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Restrict this query to the vectors in `candidates`.
    #[must_use]
    pub const fn with_candidates(mut self, candidates: &'a CandidateSet) -> Self {
        self.candidates = Some(candidates);
        self
    }

    /// Query vector.
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    /// Inclusive lower distance bound.
    pub const fn d_min(&self) -> f32 {
        self.d_min
    }

    /// Inclusive upper distance bound.
    pub const fn d_max(&self) -> f32 {
        self.d_max
    }

    /// Optional per-dimension weighted-L2 coefficients.
    pub const fn weights(&self) -> Option<&'a [f32]> {
        self.weights
    }

    /// Optional set of vector IDs eligible to match.
    pub const fn candidates(&self) -> Option<&'a CandidateSet> {
        self.candidates
    }
}

/// A disk query: find all vectors whose query distance to `query`
/// is at most `d_max` (i.e. the full disk/ball of radius `d_max`).
///
/// This is equivalent to a [`RangeQuery`] with `d_min = 0`.
pub struct DiskQuery<'a> {
    /// The query vector. Must have the database dimension and finite values.
    pub(crate) query: &'a [f32],
    /// Inclusive radius. Must be finite and non-negative.
    pub(crate) d_max: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
    /// Optional set of vector IDs eligible to match.
    pub(crate) candidates: Option<&'a CandidateSet>,
}

impl<'a> DiskQuery<'a> {
    /// Create an unweighted disk query.
    pub const fn new(query: &'a [f32], d_max: f32) -> Self {
        Self {
            query,
            d_max,
            weights: None,
            candidates: None,
        }
    }

    /// Use squared weighted L2 distance for this query.
    ///
    /// Each value is the final coefficient in
    /// `sum(weights[i] * (x[i] - query[i])^2)`. Weights are not squared by
    /// ring-db. They are validated when the query is executed.
    #[must_use]
    pub const fn with_weights(mut self, weights: &'a [f32]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Restrict this query to the vectors in `candidates`.
    #[must_use]
    pub const fn with_candidates(mut self, candidates: &'a CandidateSet) -> Self {
        self.candidates = Some(candidates);
        self
    }

    /// Query vector.
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    /// Inclusive upper distance bound.
    pub const fn d_max(&self) -> f32 {
        self.d_max
    }

    /// Optional per-dimension weighted-L2 coefficients.
    pub const fn weights(&self) -> Option<&'a [f32]> {
        self.weights
    }

    /// Optional set of vector IDs eligible to match.
    pub const fn candidates(&self) -> Option<&'a CandidateSet> {
        self.candidates
    }
}

/// A disk-intersection query: find all vectors inside every disk.
///
/// Each disk is a [`DiskQuery`] containing a center vector and an inclusive
/// radius. The query must contain at least one disk. Returned [`Hit::dist_sq`]
/// values are measured using the first disk's metric.
pub struct DiskIntersectionQuery<'a> {
    /// Disks that all returned vectors must satisfy.
    pub(crate) disks: &'a [DiskQuery<'a>],
    /// Optional set of vector IDs eligible to match.
    pub(crate) candidates: Option<&'a CandidateSet>,
}

impl<'a> DiskIntersectionQuery<'a> {
    /// Create a disk-intersection query.
    pub const fn new(disks: &'a [DiskQuery<'a>]) -> Self {
        Self {
            disks,
            candidates: None,
        }
    }

    /// Restrict this query to the vectors in `candidates`.
    #[must_use]
    pub const fn with_candidates(mut self, candidates: &'a CandidateSet) -> Self {
        self.candidates = Some(candidates);
        self
    }

    /// Disks that all returned vectors must satisfy.
    pub const fn disks(&self) -> &'a [DiskQuery<'a>] {
        self.disks
    }

    /// Optional set of vector IDs eligible to match.
    pub const fn candidates(&self) -> Option<&'a CandidateSet> {
        self.candidates
    }
}

/// An exact nearest-neighbor query.
///
/// Distances are evaluated exactly rather than through an approximate index.
/// The query returns up to `k` eligible vectors ordered by ascending
/// `(distance squared, vector ID)`. The vector ID tie-break makes results
/// deterministic when multiple vectors have the same distance. A `k` of zero
/// returns no hits.
pub struct KnnQuery<'a> {
    /// The query vector. Must have the database dimension and finite values.
    pub(crate) query: &'a [f32],
    /// Maximum number of nearest neighbors to return.
    pub(crate) k: usize,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
    /// Optional set of vector IDs eligible to match.
    pub(crate) candidates: Option<&'a CandidateSet>,
}

impl<'a> KnnQuery<'a> {
    /// Create an unweighted exact nearest-neighbor query.
    pub const fn new(query: &'a [f32], k: usize) -> Self {
        Self {
            query,
            k,
            weights: None,
            candidates: None,
        }
    }

    /// Use squared weighted L2 distance for this query.
    ///
    /// Each value is the final coefficient in
    /// `sum(weights[i] * (x[i] - query[i])^2)`. Weights are not squared by
    /// ring-db. They are validated when the query is executed.
    #[must_use]
    pub const fn with_weights(mut self, weights: &'a [f32]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Restrict this query to the vectors in `candidates`.
    #[must_use]
    pub const fn with_candidates(mut self, candidates: &'a CandidateSet) -> Self {
        self.candidates = Some(candidates);
        self
    }

    /// Query vector.
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    /// Maximum number of nearest neighbors to return.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Optional per-dimension weighted-L2 coefficients.
    pub const fn weights(&self) -> Option<&'a [f32]> {
        self.weights
    }

    /// Optional set of vector IDs eligible to match.
    pub const fn candidates(&self) -> Option<&'a CandidateSet> {
        self.candidates
    }
}

/// Result of a vector query.
pub struct QueryResult {
    /// All matching vectors together with their squared distances.
    pub hits: Vec<Hit>,
    /// Name of the backend that executed the query (e.g. `"cpu"` or `"cuda"`).
    pub backend_used: &'static str,
    /// Wall-clock time for the query (excluding dataset upload).
    pub elapsed: Duration,
}

impl QueryResult {
    /// Convenience: collect just the IDs from `hits` into a new `Vec<u32>`.
    ///
    /// Useful when calling `fetch_payloads`, which takes `&[u32]`.
    pub fn ids(&self) -> Vec<u32> {
        self.hits.iter().map(|h| h.id).collect()
    }
}
