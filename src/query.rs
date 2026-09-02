use std::time::Duration;

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
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub(crate) query: &'a [f32],
    /// Target distance (centre of the ring).
    pub(crate) d: f32,
    /// Half-width of the ring.
    pub(crate) lambda: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
}

impl<'a> RingQuery<'a> {
    /// Create an unweighted ring query.
    pub const fn new(query: &'a [f32], d: f32, lambda: f32) -> Self {
        Self {
            query,
            d,
            lambda,
            weights: None,
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
}

/// A range query: find all vectors whose query distance to `query`
/// lies within `[d_min, d_max]`.
pub struct RangeQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub(crate) query: &'a [f32],
    /// Lower bound of the distance interval (inclusive). Must be ≥ 0.
    pub(crate) d_min: f32,
    /// Upper bound of the distance interval (inclusive). Must be ≥ `d_min`.
    pub(crate) d_max: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
}

impl<'a> RangeQuery<'a> {
    /// Create an unweighted range query.
    pub const fn new(query: &'a [f32], d_min: f32, d_max: f32) -> Self {
        Self {
            query,
            d_min,
            d_max,
            weights: None,
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
}

/// A disk query: find all vectors whose query distance to `query`
/// is at most `d_max` (i.e. the full disk/ball of radius `d_max`).
///
/// This is equivalent to a [`RangeQuery`] with `d_min = 0`.
pub struct DiskQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub(crate) query: &'a [f32],
    /// Radius of the disk (inclusive upper bound on distance). Must be ≥ 0.
    pub(crate) d_max: f32,
    /// Optional final per-dimension coefficients for weighted L2 distance.
    pub(crate) weights: Option<&'a [f32]>,
}

impl<'a> DiskQuery<'a> {
    /// Create an unweighted disk query.
    pub const fn new(query: &'a [f32], d_max: f32) -> Self {
        Self {
            query,
            d_max,
            weights: None,
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
}

/// A disk-intersection query: find all vectors inside every disk.
///
/// Each disk is a [`DiskQuery`] containing a center vector and an inclusive
/// radius. The query must contain at least one disk. Returned [`Hit::dist_sq`]
/// values are measured using the first disk's metric.
pub struct DiskIntersectionQuery<'a> {
    /// Disks that all returned vectors must satisfy.
    pub(crate) disks: &'a [DiskQuery<'a>],
}

impl<'a> DiskIntersectionQuery<'a> {
    /// Create a disk-intersection query.
    pub const fn new(disks: &'a [DiskQuery<'a>]) -> Self {
        Self { disks }
    }

    /// Disks that all returned vectors must satisfy.
    pub const fn disks(&self) -> &'a [DiskQuery<'a>] {
        self.disks
    }
}

/// Result of a ring/range/disk query.
pub struct QueryResult {
    /// All matching vectors together with their squared distances.
    pub hits: Vec<Hit>,
    /// Name of the backend that executed the query (e.g. `"cpu"`, `"wgpu"`, `"cuda"`).
    pub backend_used: &'static str,
    /// Wall-clock time for the query (excluding dataset upload).
    pub elapsed: Duration,
}

impl QueryResult {
    /// Convenience: collect just the IDs from `hits` into a new `Vec<u32>`.
    ///
    /// Useful when calling `fetch_payloads` or `fetch_pods`, which take `&[u32]`.
    pub fn ids(&self) -> Vec<u32> {
        self.hits.iter().map(|h| h.id).collect()
    }
}
